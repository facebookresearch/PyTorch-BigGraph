#!/usr/bin/env python3

def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path='hdfs://<entity_path>>', # set entity_path
        edge_paths=['hdfs://<edge_path>'], # set edge_path
        checkpoint_path='hdfs://<checkpoint_path>', # set checkpoint_path
        # Graph structure
        entities={"all": {"num_partitions": 20}},
        relations=[
            {
                "name": "all_edges",
                "lhs": "all",
                "rhs": "all",
                "operator": "complex_diagonal",
            }
        ],
        dynamic_relations=True,
        verbose=1,
        # Scoring model
        dimension=100,
        batch_size=1000,
        workers=10,
        global_emb=False,
        # Training
        num_epochs=25,
        num_machines=10,
        num_uniform_negs=100,
        num_batch_negs=50,
        comparator='cos',
        loss_fn='softmax',
        distributed_init_method='env://',
        lr=0.02,
        eval_fraction=0.01  # to reproduce results we need to use all training data
    )

    return config

