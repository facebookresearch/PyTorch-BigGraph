.. _data-model:

Data model
==========

PBG operates on directed multi-relation multigraphs, whose vertices are called **entities**.
Each **edge** connects a source to a destination entity, which are respectively called its
**left-** and **right-hand side** (shortened to **LHS** and **RHS**). Multiple edges between
the same pair of entities are allowed. Loops, i.e., edges whose left- and right- hand sides
are the same, are allowed as well.

Each entity is of a certain **entity type** (one and only one type per entity).
Thus, the types partition all the entities into disjoint groups. Similarly, each
edge also belongs to exactly one **relation type**. All edges of a given
relation type must have all their left-hand side entities of the same entity
type and, similarly, all their right-hand side entities of the same entity type
(possibly a different entity type than the left-hand side one). This property
means that each relation type has a left-hand side entity type and a right-hand
side entity type.

.. figure:: _static/graph_1.svg
    :figwidth: 50 %
    :alt: a graph with three entity types and three relation types

    In this graph, there are 14 entities: 5 of the red entity type, 6 of the
    yellow entity type and 3 of the blue entity type; there are also 12 edges:
    6 of the orange relation type (between red and yellow entities), 3 of the
    purple entity type (between red and blue entities) and 3 of the green entity
    type (between yellow and blue entities).

In order for torchbiggraph to operate on large-scale graphs, the graph is broken
up into small pieces, on which training can happen in a distributed manner. This
is first achieved by further splitting the entities of each type into a certain
number of subsets, called **partitions**. Then, for each relation type, its
edges are divided into **buckets**: for each pair of partitions (one from the
left- and one from the right-hand side entity types for that relation type)
a bucket is created, which contains the edges of that type whose left- and
right-hand side entities are in those partitions.

.. figure:: _static/graph_2.svg
    :figwidth: 50 %
    :alt: the graph from before, partitioned and with only one bucket visible

    This graph shows a possible partition of the entities, with red having 2
    partitions, yellow having 3, and blue having only one (hence blue is
    unpartitioned). The edges displayed are those of the orange bucket between
    the second red partition and the second yellow partition.

.. note::
    For technical reasons, at the current state all entity types must be divided
    into the same number of partitions (except unpartitioned entities). In
    numpy-speak, it means that the number of partitions of all entities must
    be broadcastable to the same value.

An entity is identified by its type, its partition and its index within the
partition (indices must be contiguous, meaning that if there are :math:`N`
entities in a type's partition, their indices must be between 1 and :math:`N`
inclusive). An edge is identified by its type, its bucket (i.e., the partitions
of its left- and right-hand side entity types) and the indices of its left- and
right-hand side entities in their respective partitions. An edge doesn't have
to specify its left- and right-hand side entity types, because they are implicit
in the edge's relation type.
