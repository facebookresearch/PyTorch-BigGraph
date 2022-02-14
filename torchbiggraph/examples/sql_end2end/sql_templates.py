QUERY_MAKE_ID2PART_TBL = """
DROP TABLE IF EXISTS tmp_{type}_id2part
;

create temporary table tmp_{type}_id2part as
    select id, abs(random()) % {nparts} as part
    from (
        select distinct source_id as id from edges where source_type='{type}'
        union
        select distinct destination_id as id from edges where destination_type='{type}'
    )

;
"""

PARTITIONED_MAPPED_ENTITIES = """
DROP TABLE IF EXISTS tmp_{type}_ids_map_{n}
;

create table tmp_{type}_ids_map_{n} as
select 
    f.id
    , f.part
    , '{type}' as type
    , (ROW_NUMBER() OVER(ORDER BY f.id)) - 1 as graph_id
from tmp_{type}_id2part f
where f.part = {n}
order by 2 desc, 1 asc
;
"""

REMAP_RELNS = """
DROP TABLE IF EXISTS tmp_reln_map
;

create table tmp_reln_map as
select f.rel as id, source_type, destination_type, (ROW_NUMBER() OVER(ORDER BY f.rel)) - 1 as graph_id
from (
    select distinct rel, source_type, destination_type
    from edges
) f
"""

EDGELIST_CTE_MAPPER = """
    select lhs.graph_id as source_id, rel.graph_id as rel_id, rhs.graph_id as destination_id
    from edges g
    join tmp_reln_map rel on (rel.id = g.rel)
    join tmp_{lhs_type}_ids_map_{i} lhs on (
        lhs.id = g.source_id and
        g.source_type = rel.source_type and
        lhs.type = g.source_type
    )
    join tmp_{rhs_type}_ids_map_{j} rhs on (
        rhs.id = g.destination_id and
        g.destination_type = rel.destination_type and
        rhs.type = g.destination_type
    )
    where g.rel = '{rel_name}'
"""

EDGES_PARTITIONED = """
DROP TABLE IF EXISTS tmp_edges_{i}_{j}
;

create table tmp_edges_{i}_{j} as
{ctes}
select *
from (
{tables}
)
;
"""