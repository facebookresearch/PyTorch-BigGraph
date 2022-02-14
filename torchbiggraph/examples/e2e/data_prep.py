import itertools
import h5py
import json
import logging
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import sqlite3

from config_template import CONFIG_TEMPLATE
from sql_templates import (
    edges_partitioned,
    edgelist_cte_mapper,
    remap_relns,
    partitioned_mapped_entities,
    QUERY_MAKE_ID2PART_TBL
)
import sys
import time

logging.basicConfig(
    format='%(process)d-%(levelname)s-%(message)s',
    stream = sys.stdout,
    level=logging.DEBUG
)

def remap_relationships(conn):
    """
    A function to remap relationships using SQL queries.
    """
    logging.info("Remapping relationships")
    start = time.time()
    logging.debug(f"Running query: {remap_relns}\n")
    conn.executescript(remap_relns)

    query = """
    select *
    from tmp_reln_map
    """
    logging.debug(f"Running query: {query}\n")
    rels = pd.read_sql_query(query, conn)
    end = time.time()
    logging.info(f"Remapped relationships in {end - start}s")
    return rels


def remap_entities(conn, entity2partitions):
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
        query = QUERY_MAKE_ID2PART_TBL.format(type=entity, nparts=npartitions)

        for i in range(npartitions):
            query += partitioned_mapped_entities.format(type=entity, n=i)
        logging.debug(f"Running query: {query}")
        conn.executescript(query)
    end = time.time()
    logging.info(f"Remapped entities in {end - start}s")


def generate_ctes(lhs_part, rhs_part, rels, entity2partitions):
    """
    This function generates the sub-table CTEs that help us generate
    the completed edgelist.
    """
    nctes = 0
    ctes = """
    with cte_0 as (
    """
    first = True
    for _ , r in rels.iterrows():
        if lhs_part >= entity2partitions[r['source_type']]:
            continue
        if rhs_part >= entity2partitions[r['destination_type']]:
            continue
        if not first:
            ctes += f", cte_{nctes} as ("
        ctes += edgelist_cte_mapper.format(
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


def generate_unions(nctes):
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


def remap_edges(conn, rels, entity2partitions):
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
    NPARTS = max(entity2partitions.values())
    for lhs_part in range(NPARTS):
        for rhs_part in range(NPARTS):
            nctes, ctes = generate_ctes(lhs_part, rhs_part, rels, entity2partitions)
            subquery = generate_unions(nctes)
            query += edges_partitioned.format(
                i = lhs_part,
                j = rhs_part,
                ctes=ctes,
                tables=subquery
            )

    logging.debug(f"Running query: {query}")
    conn.executescript(query)

    logging.debug("Confirming that we didn't drop any edges.")
    nentities_postmap = 0
    for lhs_part in range(NPARTS):
        for rhs_part in range(NPARTS):
            nentities_postmap += conn.execute(f"""
            select count(*) from tmp_edges_{lhs_part}_{rhs_part}
            """).fetchall()[0][0]
    
    if nentities_postmap != nentities_premap:
        logging.warning("DROPPED EDGES DURING REMAPPING.")
        logging.warning(f"We started with {nentities_premap} and finished with {nentities_postmap}")

    end = time.time()
    logging.info(f"Remapped edges in {end - start}s")


def load_edges(fname, conn):
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


def write_relations(outdir, rels, conn):
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


def write_single_bucket(work_packet):
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


def write_all_buckets(outdir, lhs_parts, rhs_parts, conn):
    """
    A function to write out all edge-lists in the format
    that PyTorch BigGraph expects.
    """
    logging.info(f"Writing edges, {lhs_parts}, {rhs_parts}")
    start = time.time()

    # I would write these using multiprocessing but SQLite connections
    # aren't pickelable, and I'd like to keep this simple
    worklist = list(itertools.product(range(lhs_parts), range(rhs_parts), ['training_data'], [conn]))
    for w in worklist:
        write_single_bucket(w)

    end = time.time()
    logging.info(f"Wrote edges in {end - start}s")
    

def write_entities(outdir, entity2partitions, conn):
    """
    A function to write out all of the training relevant
    entity information that PyTorch BigGraph expects
    """
    logging.info("Writing entites for training")
    start = time.time()
    for entity_type, nparts in entity2partitions.items():
        for i in range(nparts):
            query = f"""
                select count(*)
                from {entity_type}_ids_map_{i}
            """
            sz = conn.execute(query).fetchall()[0][0]
            with open(f'{outdir}/entity_count_{entity_type}_id_{i}.txt', mode='w') as f:
                f.write(f"{sz}\n")
    end = time.time()
    logging.info(f"Wrote entites in {end - start}s")


def write_training_data(outdir, rels, entity2partitions, conn):
    """
    A function to write out all of the training relevant
    information that PyTorch BigGraph expects
    """
    lhs_parts = 1
    rhs_parts = 1
    for i, r in rels.iterrows():
        if entity2partitions[r['source_type']] > lhs_parts:
            lhs_parts = entity2partitions[r['source_type']]
        if entity2partitions[r['destination_type']] > rhs_parts:
            rhs_parts = entity2partitions[r['destination_type']]

    write_relations(outdir, rels, conn)
    write_all_buckets(rels, lhs_parts, rhs_parts, conn)
    write_entities(outdir, entity2partitions, conn)


def write_rels_dict(rels):
  my_rels = ""
  for _, row in rels.sort_values(by="graph_id").iterrows():
    r = "{"
    r += f"'name': '{row['id']}', 'lhs': '{row['source_type']}', 'rhs': '{row['destination_type']}', 'operator': op"
    r += "},\n"
    my_rels += r
  return my_rels


def write_entities_dict(entity2partitions):
    my_entities = "{\n"
    for name, part in entity2partitions.items():
        my_entities += '{ "{name}": {"num_partitions": {part}} }'.format(name=name, part=part)
    my_entities += "}\n"
    return my_entities


def write_config(rels, entity2partitions, config_out, train_out, model_out):
    with open(config_out, mode='w') as f:
        f.write(
            CONFIG_TEMPLATE.format(
                RELN_DICT=write_rels_dict(rels),
                ENTITIES_DICT=write_entities_dict(entity2partitions),
                TRAINING_DIR=train_out,
                MODEL_PATH=model_out,
            )
        )


def compute_memory_usage(entity2partitions, conn, NDIM=200):
    nentities = 0
    for _type, parts in entity2partitions.items():
        ntype = 0
        for i in range(parts):
            query = f"""
                select count(*) as cnt
                from `tmp_{_type}_ids_map_{i}`
                """
            ntype = max(ntype, conn.executequery(query).fetchall()[0][0])
        nentities += ntype

    mem = 1.5 * nentities * NDIM * 8 / 1024 / 1024 / 1024
    logging.info(f"I need {mem} GBs of ram for embedding table for {NDIM} Dimensions")


def main(NPARTS=2, edge_file_name='edges.csv', outdir='training_data/', modeldir='model/', config_dir='.'):
    conn = sqlite3.connect("citationv2.db")
    load_edges(edge_file_name, conn)

    entity2partitions = {
        'paper': NPARTS,
        'year': 1,
    }

    rels = remap_relationships(conn)
    remap_entities(conn, entity2partitions)
    remap_edges(conn, rels, entity2partitions)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    out = Path(outdir)
    write_training_data(out, rels, entity2partitions, conn)
    write_config(rels, entity2partitions, config_dir, out, modeldir)


if __name__ == '__main__':
    main()