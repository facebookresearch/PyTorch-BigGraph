type_tmp_table = """
DROP TABLE IF EXISTS {type}_id2part
;

create temporary table {type}_id2part as
    select id, abs(random()) % {nparts} as part
    from (
        select distinct source_id as id from edges where source_type='{type}'
        union
        select distinct destination_id as id from edges where destination_type='{type}'
    )

;
"""

partitioned_mapped_entities = """
DROP TABLE IF EXISTS {type}_ids_map_{n}
;

create table {type}_ids_map_{n} as
select 
    f.id
    , f.part
    , '{type}' as type
    , (ROW_NUMBER() OVER(ORDER BY f.id)) - 1 as graph_id
from {type}_id2part f
where f.part = {n}
order by 2 desc, 1 asc
;
"""

remap_relns = """
DROP TABLE IF EXISTS reln_map
;

create table reln_map as
select f.rel as id, source_type, destination_type, (ROW_NUMBER() OVER(ORDER BY f.rel)) - 1 as graph_id
from (
    select distinct rel, source_type, destination_type
    from edges
) f
"""

edgelist_cte_mapper = """
    select lhs.graph_id as source_id, rel.graph_id as rel_id, rhs.graph_id as destination_id
    from edges g
    join reln_map rel on (rel.id = g.rel)
    join {lhs_type}_ids_map_{i} lhs on (
        lhs.id = g.source_id and
        g.source_type = rel.source_type and
        lhs.type = g.source_type
    )
    join {rhs_type}_ids_map_{j} rhs on (
        rhs.id = g.destination_id and
        g.destination_type = rel.destination_type and
        rhs.type = g.destination_type
    )
    where g.rel = '{rel_name}'
"""

edges_partitioned = """
DROP TABLE IF EXISTS edges_{i}_{j}
;

create table edges_{i}_{j} as
{ctes}
select *
from (
{tables}
)
;
"""