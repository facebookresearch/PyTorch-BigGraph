import argparse
import logging
import sys
import torch

from torchbiggraph.train_cpu import (
    IterationManager,
    get_num_edge_chunks,
)
from torchbiggraph.graph_storages import EDGE_STORAGES, ENTITY_STORAGES
from torchbiggraph.config import ConfigFileLoader, ConfigSchema
from torchbiggraph.types import Bucket
from torchbiggraph.util import EmbeddingHolder

logger = logging.getLogger("torchbiggraph")

class Checker:
    def __init__(self, config):
        entity_storage = ENTITY_STORAGES.make_instance(config.entity_path)
        entity_counts = {}
        for entity, econf in config.entities.items():
            entity_counts[entity] = []
            for part in range(econf.num_partitions):
                entity_counts[entity].append(entity_storage.load_count(entity, part))
        self.entity_counts = entity_counts
        self.config = config
        holder = self.holder = EmbeddingHolder(config)


    def check_all_edges(self):
        num_edge_chunks = get_num_edge_chunks(self.config)

        iteration_manager = IterationManager(
            1,
            self.config.edge_paths,
            num_edge_chunks,
            iteration_idx=0,
        )
        edge_storage = EDGE_STORAGES.make_instance(iteration_manager.edge_path)

        for _, _, edge_chunk_idx in iteration_manager:
            for lhs in range(self.holder.nparts_lhs):
                for rhs in range(self.holder.nparts_rhs):
                    cur_b = Bucket(lhs, rhs)
                    logging.info(f"Checking edge chunk: {edge_chunk_idx} for edges_{cur_b.lhs}_{cur_b.rhs}.h5")
                    edges = edge_storage.load_chunk_of_edges(
                        cur_b.lhs,
                        cur_b.rhs,
                        edge_chunk_idx,
                        iteration_manager.num_edge_chunks,
                        shared=True,
                    )
                    self.check_edge_chunk(cur_b, edges)

    def check_edge_chunk(self, cur_b, edges):
        rhs = edges.rhs.to_tensor()
        lhs = edges.lhs.to_tensor()
        rel_lhs_entity_counts = torch.tensor(
            [self.entity_counts[r.lhs][cur_b.lhs] for r in self.config.relations]
        )
        #Check LHS         
        edge_lhs_entity_count = rel_lhs_entity_counts[edges.rel]

        if any(lhs >= edge_lhs_entity_count):
            _, worst_edge_idx = (lhs - edge_lhs_entity_count).max(0)
            raise RuntimeError(f"edge {worst_edge_idx} has LHS entity of "
                                f"{lhs[worst_edge_idx]} but rel "
                                f"{edges.rel[worst_edge_idx]} only has "
                                f"{edge_lhs_entity_count[worst_edge_idx]} "
                                "entities "
                                f" with r.name: {self.config.relations[edges.rel[worst_edge_idx]].name}. "
                                "Preprocessing bug?")
        #Check RHS
        rel_rhs_entity_counts = torch.tensor(
            [self.entity_counts[r.rhs][cur_b.rhs] for r in self.config.relations]
        )
        edge_rhs_entity_count = rel_rhs_entity_counts[edges.rel]
        if any(rhs >= edge_rhs_entity_count):
            _, worst_edge_idx = (rhs - edge_rhs_entity_count).max(0)
            raise RuntimeError(f"edge {worst_edge_idx} has RHS entity of "
                                f"{rhs[worst_edge_idx]} but rel "
                                f"{edges.rel[worst_edge_idx]} only has "
                                f"{edge_rhs_entity_count[worst_edge_idx]} "
                                "entities "
                                f" with r.name: {self.config.relations[edges.rel[worst_edge_idx]].name}. "
                                "Preprocessing bug?")       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("-p", "--param", action="append", nargs="*")
    opt = parser.parse_args()

    loader = ConfigFileLoader()
    config = loader.load_config(opt.config, opt.param)


    Checker(config).check_all_edges()
    logging.info("Found no errors in the input directory")