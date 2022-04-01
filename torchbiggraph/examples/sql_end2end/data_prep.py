import argparse
import copy
import itertools
import h5py
import json
import logging
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import sqlite3
from sqlite3 import Connection
from typing import Dict, Tuple

from config_template import CONFIG_TEMPLATE
from sql_templates import (
    EDGES_PARTITIONED,
    EDGELIST_CTE_MAPPER,
    REMAP_RELNS,
    PARTITIONED_MAPPED_ENTITIES,
    QUERY_MAKE_ID2PART_TBL
)
import sys
import time

DEFAULT_CFG = dict(
    # Scoring model
    dimension=200,
    comparator="dot",
    loss_fn="softmax",
    # Training
    num_epochs=50,
    num_uniform_negs=1000,
    num_batch_negs=1000,
    batch_size=150_000,
    lr=0.05,
    regularization_coef=1e-3,
    num_gpus=2,
    eval_fraction=0,
    # io
    checkpoint_path='model',
    entity_path='training_data',
    edge_paths=[
        'training_data'
    ],
)


logging.basicConfig(
    format='%(process)d-%(levelname)s-%(message)s',
    stream = sys.stdout,
    level=logging.DEBUG
)

def remap_relationships(conn: Connection) -> pd.DataFrame:
    """
    A function to remap relationships using SQL queries.
    """
    logging.info("Remapping relationships")
    start = time.time()
    logging.debug(f"Running query: {REMAP_RELNS}\n")
    conn.executescript(REMAP_RELNS)

    query = """
    select *
    from tmp_reln_map
    """
    logging.debug(f"Running query: {query}\n")
    rels = pd.read_sql_query(query, conn)
    end = time.time()
    logging.info(f"Remapped relationships in {end - start}s")
    return rels


def remap_entities(conn: Connection, entity2partitions:  Dict[str, Dict[str, int]]) -> None:
    """
    A function to remap entities with partitioning using SQL queries.

    This function is complicated because the partitions have to be
    constructed first, and then we have to generate ordinal mappings of
    entity ids. These mappings will be used to generate buckets of edges
    for training and then for mapping our edges back to their original
    ids for use in downstream tasks.
    """
    logging.info("Remapping entities")
    start=time.time()
    query = ""
    for entity, npartitions in entity2partitions.items():
        query = QUERY_MAKE_ID2PART_TBL.format(type=entity, nparts=npartitions['num_partitions'])

        for i in range(npartitions['num_partitions']):
            query += PARTITIONED_MAPPED_ENTITIES.format(type=entity, n=i)
        logging.debug(f"Running query: {query}")
        conn.executescript(query)
    end = time.time()
    logging.info(f"Remapped entities in {end - start}s")


def generate_ctes(lhs_part: int, rhs_part: int, rels: int, entity2partitions:  Dict[str, Dict[str, int]]) -> Tuple[int, str]:
    """
    This function generates the sub-table Common Table Expressions (CTES)
    that help us generate the completed edgelist.
    """
    nctes = 0
    ctes = """
    with cte_0 as (
    """
    first = True
    for _ , r in rels.iterrows():
        if lhs_part >= entity2partitions[r['source_type']]['num_partitions']:
            continue
        if rhs_part >= entity2partitions[r['destination_type']]['num_partitions']:
            continue
        if not first:
            ctes += f", cte_{nctes} as ("
        ctes += EDGELIST_CTE_MAPPER.format(
            rel_name=r['id'],
            lhs_type=r['source_type'],
            rhs_type=r['destination_type'],
            i = lhs_part,
            j = rhs_part,
        )
        ctes += ")"

        nctes += 1
        first = False
    return nctes, ctes


def generate_unions(nctes: int) -> str:
    """
    This function is just a helper function for
    generating the final edge list tables.
    """
    subquery = ""
    first = True
    for i in range(nctes):
        if not first:
            subquery += "\tunion\n"
        subquery += f"\tselect * from cte_{i}\n"
        first = False
    return subquery


def remap_edges(conn: Connection, rels: pd.DataFrame, entity2partitions:  Dict[str, Dict[str, int]]) -> None:
    """
    A function to remap all edges to ordinal IDs
    according to their type.
    """
    logging.info("Remapping edges")
    start = time.time()

    nentities_premap = conn.execute("""
    select count(*) from edges
    ;
    """).fetchall()[0][0]

    query = ""
    nparts = max([n['num_partitions'] for _, n in entity2partitions.items()])
    for lhs_part in range(nparts):
        for rhs_part in range(nparts):
            nctes, ctes = generate_ctes(lhs_part, rhs_part, rels, entity2partitions)
            subquery = generate_unions(nctes)
            query += EDGES_PARTITIONED.format(
                i = lhs_part,
                j = rhs_part,
                ctes=ctes,
                tables=subquery
            )

    logging.debug(f"Running query: {query}")
    conn.executescript(query)

    logging.debug("Confirming that we didn't drop any edges.")
    nentities_postmap = 0
    for lhs_part in range(nparts):
        for rhs_part in range(nparts):
            nentities_postmap += conn.execute(f"""
            select count(*) from tmp_edges_{lhs_part}_{rhs_part}
            """).fetchall()[0][0]
    
    if nentities_postmap != nentities_premap:
        logging.warning("DROPPED EDGES DURING REMAPPING.")
        logging.warning(f"We started with {nentities_premap} and finished with {nentities_postmap}")

    end = time.time()
    logging.info(f"Remapped edges in {end - start}s")


def load_edges(fname: str, conn: Connection) -> None:
    """
    A simple function to load the edges into the SQL table. It is
    assumed that we will have a file of the form:
    | source_id | source_type | relationship_name | destination_id | destination_type |.

    For production applications you wouldn't use this step; it's just for our example.    
    """
    logging.info("Loading edges")
    start = time.time()
    cur = conn.cursor()
    cur.executescript("""
    DROP TABLE IF EXISTS edges
    ;

    CREATE TABLE edges (
        source_id INTEGER,
        source_type TEXT,
        destination_id INTEGER,
        destination_type TEXT,
        rel TEXT
    )
    """)

    edges = pd.read_csv(fname)
    edges.to_sql('edges', conn, if_exists='append', index=False)
    end = time.time()
    logging.info(f"Loading edges in {end - start}s")


def write_relations(outdir: Path, rels: pd.DataFrame, conn: Connection) -> None:
    """
    A simple function to write the relevant relationship information out
    for training.
    """
    logging.info("Writing relations for training")
    start = time.time()

    out = rels.sort_values('graph_id')['id'].to_list()
    with open(f'{outdir}/dynamic_rel_names.json', mode='w') as f:
        json.dump(out, f, indent=4)
    end = time.time()
    logging.info(f"Wrote relations in {end - start}s")


def write_single_bucket(work_packet: Tuple[int, int, Path, Connection]) -> None:
    """
    A function to write out a single edge-lists in the format that
    PyTorch BigGraph expects.

    The work packet is expected to come contain information about
    the lhs and rhs partitions for these edges, the directory
    where we should put this information, and the database
    connection that we should use.
    """
    lhs_part, rhs_part, outdir, conn = work_packet
    query = f"""
    select *
    from tmp_edges_{lhs_part}_{rhs_part}
    ;
    """
    df = pd.read_sql_query(query, conn)
    print(query)
    out_name = f'{outdir}/edges_{lhs_part}_{rhs_part}.h5'
    with h5py.File(out_name, mode='w') as f:
        # we need this for https://github.com/facebookresearch/PyTorch-BigGraph/blob/main/torchbiggraph/graph_storages.py#L400
        f.attrs['format_version'] = 1
        for dset, colname in [('lhs', 'source_id'), ('rhs', 'destination_id'), ('rel', 'rel_id')]:
            f.create_dataset(dset, dtype='i', shape=(len(df),), maxshape=(None, ))
            f[dset][0 : len(df)] = df[colname].tolist()


def write_all_buckets(outdir: Path, lhs_parts: int, rhs_parts: int, conn: Connection) -> None:
    """
    A function to write out all edge-lists in the format
    that PyTorch BigGraph expects.
    """
    logging.info(f"Writing edges, {lhs_parts}, {rhs_parts}")
    start = time.time()

    # I would write these using multiprocessing but SQLite connections
    # aren't pickelable, and I'd like to keep this simple
    worklist = list(itertools.product(range(lhs_parts), range(rhs_parts), [outdir], [conn]))
    for w in worklist:
        write_single_bucket(w)

    end = time.time()
    logging.info(f"Wrote edges in {end - start}s")
    

def write_entities(
    outdir: Path,
    entity2partitions:  Dict[str, Dict[str, int]],
    conn: Connection) -> None:
    """
    A function to write out all of the training relevant
    entity information that PyTorch BigGraph expects
    """
    logging.info("Writing entites for training")
    start = time.time()

    for entity_type, nparts in entity2partitions.items():
        for i in range(nparts['num_partitions']):
            query = f"""
                select count(*)
                from tmp_{entity_type}_ids_map_{i}
            """
            sz = conn.execute(query).fetchall()[0][0]
            with open(f'{outdir}/entity_count_{entity_type}_{i}.txt', mode='w') as f:
                f.write(f"{sz}\n")
    end = time.time()
    logging.info(f"Wrote entites in {end - start}s")


def write_training_data(
    outdir: Path,
    rels: pd.DataFrame,
    entity2partitions:  Dict[str, Dict[str, int]],
    conn: Connection
    ) -> None:
    """
    A function to write out all of the training relevant
    information that PyTorch BigGraph expects
    """
    lhs_parts = 1
    rhs_parts = 1
    for _, r in rels.iterrows():
        if entity2partitions[r['source_type']]['num_partitions'] > lhs_parts:
            lhs_parts = entity2partitions[r['source_type']]['num_partitions']
        if entity2partitions[r['destination_type']]['num_partitions'] > rhs_parts:
            rhs_parts = entity2partitions[r['destination_type']]['num_partitions']

    print("LHS_PARTS: ", lhs_parts)
    print("RHS_PARTS: ", rhs_parts)

    write_relations(outdir, rels, conn)
    write_all_buckets(outdir, lhs_parts, rhs_parts, conn)
    write_entities(outdir, entity2partitions, conn)


def write_config(
    rels: pd.DataFrame,
    entity2partitions: Dict[str, Dict[str, int]],
    config_out: Path,
    train_out: Path,
    model_out: Path,
    ndim: int = 200,
    ngpus:int = 2
    ) -> None:
    outname = config_out / 'config.py'
    rels['operator'] = 'translation'
    rels = rels.rename({'id': 'name', 'source_type': 'lhs', 'destination_type': 'rhs'}, axis=1)

    cfg = copy.deepcopy(DEFAULT_CFG)
    cfg['edge_paths'] = [ train_out.as_posix() ]
    cfg['entity_path'] = train_out.as_posix()
    cfg['checkpoint_path'] = model_out.as_posix()
    cfg['entites'] = entity2partitions
    cfg['relations'] = rels[['name', 'lhs', 'rhs', 'operator']].to_dict(orient='records')

    with open(outname, mode='w') as f:
        f.write(
            f"def get_torchbiggraph_config():\n\treturn {json.dumps(cfg, indent=4)}\n"
        )


def compute_memory_usage(
    entity2partitions: Dict[str, Dict[str, int]],
    conn: Connection,
    ndim: int) -> None:
    nentities = 0
    for _type, parts in entity2partitions.items():
        ntype = 0
        for i in range(parts['num_partitions']):
            query = f"""
                select count(*) as cnt
                from `tmp_{_type}_ids_map_{i}`
                """
            res = pd.read_sql_query(query, conn)
            res = conn.execute(query).fetchall()
            ntype = max(ntype, res[0][0])
        nentities += ntype

    # 1.2 here is an empirical safety factor.
    mem = 1.2 * nentities * ndim * 8 / 1024 / 1024 / 1024
    logging.info(f"I need {mem} GBs of ram for embedding table for {ndim} Dimensions")


def main(
    nparts: int = 1,
    edge_file_name: str = 'edges.csv',
    outdir: Path = Path('training_data/'),
    modeldir: Path = Path('model/'),
    config_dir: Path = Path(''),
    dbname: str = 'citationv2.db') -> None:
    conn = sqlite3.connect(dbname)
    # load_edges(edge_file_name, conn)

    entity2partitions = {
        'paper': {'num_partitions': nparts},
        'year': {'num_partitions': 1},
    }

    rels = remap_relationships(conn)
    # remap_entities(conn, entity2partitions)
    # remap_edges(conn, rels, entity2partitions)

    outdir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    modeldir.mkdir(parents=True, exist_ok=True)

    # write_training_data(outdir, rels, entity2partitions, conn)
    write_config(rels, entity2partitions, config_dir, outdir, modeldir)
    compute_memory_usage(entity2partitions, conn, 200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-npart", help="The number of partitions to split the paper_ids into", type=int)
    parser.add_argument("-e", help="The edges file to load in")
    parser.add_argument("-o", help="The directory where the training data should be stored", required=False)
    parser.add_argument("-m", help="The directory where the model artifacts should be stored", required=False)
    parser.add_argument("-c", help="The location where the generated config file will be stored", required=False)
    opt = parser.parse_args()

    main(
        nparts=opt.npart,
        edge_file_name=opt.e,
        outdir=Path(opt.o),
        modeldir=Path(opt.m),
        config_dir=Path(opt.c),
    )