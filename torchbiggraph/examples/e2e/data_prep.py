import itertools
import h5py
import json
import logging
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import sqlite3

from sklearn.multiclass import OutputCodeClassifier
from sql_templates import (
    edges_partitioned,
    edgelist_cte_mapper,
    remap_relns,
    partitioned_mapped_entities,
    type_tmp_table
)
import sys
import time

logging.basicConfig(
    format='%(process)d-%(levelname)s-%(message)s',
    stream = sys.stdout,
    level=logging.DEBUG
)

"""
This is intended as a simple end-to-end example of how to get your data into
the format that PyTorch BigGraph expects using SQL. It's implemented in SQLite
for portability, but similar techniques scale to 100bn edges using cloud
databases such as BigQuery. This pipeline can be split into three different
components:

1. Data preparation
2. Data verification/checking
3. Training

To run the pipeline, you'll first need to download the edges.csv file,
available HERE (TODO: INSERT LINK). This graph was constructed by
taking the [ogbl-citation2](https://github.com/snap-stanford/ogb) graph, and
adding edges for both paper-citations and years-published. While this graph
might not make a huge amount of sense, it's intended to largely fulfill a
pedagogical purpose. In the data preparation stage, we first load the graph
into a SQLite database, and then we transform and partition it.
"""

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
    from reln_map
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
        query = type_tmp_table.format(type=entity, nparts=npartitions)

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
            select count(*) from edges_{lhs_part}_{rhs_part}
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
    | source_id | source_type | relationship_name | destination_id | destination_type |
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


def write_single_edge(work_packet):
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
    from edges_{lhs_part}_{rhs_part}
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


def write_edges(outdir, LHS_PARTS, RHS_PARTS, conn):
    """
    A function to write out all edge-lists in the format
    that PyTorch BigGraph expects.
    """
    logging.info(f"Writing edges, {LHS_PARTS}, {RHS_PARTS}")
    start = time.time()

    # I would write these using multiprocessing but SQLite connections
    # aren't pickelable, and I'd like to keep this simple
    worklist = list(itertools.product(range(LHS_PARTS), range(RHS_PARTS), ['training_data'], [conn]))
    for w in worklist:
        write_single_edge(w)

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
    LHS_PARTS = 1
    RHS_PARTS = 1
    for i, r in rels.iterrows():
        if entity2partitions[r['source_type']] > LHS_PARTS:
            LHS_PARTS = entity2partitions[r['source_type']]
        if entity2partitions[r['destination_type']] > RHS_PARTS:
            RHS_PARTS = entity2partitions[r['destination_type']]

    write_relations(outdir, rels, conn)
    write_edges(rels, LHS_PARTS, RHS_PARTS, conn)
    write_entities(outdir, entity2partitions, conn)


def main(NPARTS=2, edge_file_name='edges.csv', outdir='training_data/'):
    conn = sqlite3.connect("citationv2.db")
    # load_edges(edge_file_name, conn)

    entity2partitions = {
        'paper': NPARTS,
        'year': 1,
    }

    rels = remap_relationships(conn)
    # remap_entities(conn, entity2partitions)
    # remap_edges(conn, rels, entity2partitions)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    out = Path(outdir)
    write_training_data(out, rels, entity2partitions, conn)


if __name__ == '__main__':
    main()