import networkx as nx
from ....enums import LandUse

adjacency_rules_list = [
    # self adjacency
    (LandUse.RESIDENTIAL, LandUse.RESIDENTIAL),
    (LandUse.BUSINESS, LandUse.BUSINESS),
    (LandUse.RECREATION, LandUse.RECREATION),
    (LandUse.INDUSTRIAL, LandUse.INDUSTRIAL),
    (LandUse.TRANSPORT, LandUse.TRANSPORT),
    (LandUse.SPECIAL, LandUse.SPECIAL),
    (LandUse.AGRICULTURE, LandUse.AGRICULTURE),
    # recreation can be adjacent to anything
    (LandUse.RECREATION, LandUse.SPECIAL),
    (LandUse.RECREATION, LandUse.INDUSTRIAL),
    (LandUse.RECREATION, LandUse.BUSINESS),
    (LandUse.RECREATION, LandUse.AGRICULTURE),
    (LandUse.RECREATION, LandUse.TRANSPORT),
    (LandUse.RECREATION, LandUse.RESIDENTIAL),
    # residential
    (LandUse.RESIDENTIAL, LandUse.BUSINESS),
    # business
    (LandUse.BUSINESS, LandUse.INDUSTRIAL),
    (LandUse.BUSINESS, LandUse.TRANSPORT),
    # industrial
    (LandUse.INDUSTRIAL, LandUse.SPECIAL),
    (LandUse.INDUSTRIAL, LandUse.AGRICULTURE),
    (LandUse.INDUSTRIAL, LandUse.TRANSPORT),
    # transport
    (LandUse.TRANSPORT, LandUse.SPECIAL),
    (LandUse.TRANSPORT, LandUse.AGRICULTURE),
    # special
    (LandUse.SPECIAL, LandUse.AGRICULTURE),
]

ADJACENCY_RULES_GRAPH = nx.from_edgelist(adjacency_rules_list)
