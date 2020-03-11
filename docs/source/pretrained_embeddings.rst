Pre-trained embeddings
======================

For demonstration purposes and to save users their time, we provide pre-trained embeddings for
some common public datasets.

.. _wiki-data:

Wikidata
--------

`Wikidata <https://www.wikidata.org/>`_ is a well-known knowledge base, which includes the discontinued Freebase
knowledge base.

We used the so-called "truthy" dump from 2019-03-06, in the RDF NTriples format. (The original file isn't available
anymore on the Wikidata website). We used as entities all the distinct strings that appeared as either source or
target nodes in this dump: this means that entities include URLs of Wikidata entities (in the form :samp:`<http://www.wikidata.org/entity/Q{123}>`),
plain quoted strings (e.g., :samp:`"{Foo}"`), strings with language annotation (e.g., :samp:`"{Bar}"@{fr}`), dates and times, and possibly more.
Similarly, we used as relation types all the distinct strings that appeared as properties. We then filtered out entities and relation types that
appeared less than 5 times in the data dump.

The embeddings were trained with the following configuration::

    def get_torchbiggraph_config():

        config = dict(
            # I/O data
            entity_path='data/wikidata',
            edge_paths=[],
            checkpoint_path='model/wikidata',

            # Graph structure
            entities={
                'all': {'num_partitions': 1},
            },
            relations=[{
                'name': 'all_edges',
                'lhs': 'all',
                'rhs': 'all',
                'operator': 'translation',
            }],
            dynamic_relations=True,

            # Scoring model
            dimension=200,
            global_emb=False,
            comparator='dot',

            # Training
            num_epochs=4,
            num_edge_chunks=10,
            batch_size=10000,
            num_batch_negs=500,
            num_uniform_negs=500,
            loss_fn='softmax',
            lr=0.1,
            relation_lr=0.01,

            # Evaluation during training
            eval_fraction=0.001,
            eval_num_batch_negs=10000,
            eval_num_uniform_negs=0,

            # Misc
            verbose=1,
        )

        return config

The output embeddings are available in various formats:

- `wikidata_translation_v1.tsv.gz <https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1.tsv.gz>`_ (36GiB),
  a gzipped TSV (tab-separated value) file in an old format produced by ``torchbiggraph_export_to_tsv``
  (see :ref:`here <tsv-format>` for how to parse it).
- `wikidata_translation_v1_names.json.gz <https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1_names.json.gz>`_ (378MiB),
  a gzipped JSON-encoded list of all the keys in the first column of the TSV file.
- `wikidata_translation_v1_vectors.npy.gz <https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1_vectors.npy.gz>`_ (39.9GiB),
  a gzipped serialized NumPy array with the 200-dimension vectors, one for each line of the TSV file.
