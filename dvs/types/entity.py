import typing

from ner_agent import Entity as NerEntity

if typing.TYPE_CHECKING:
    from dvs.types.edge import Edge
    from dvs.types.node import Node


class Entity(NerEntity):
    document_id: str

    def to_nodes_edges(
        self,
        *,
        canonical_map: typing.Optional[typing.Dict[str, str]] = None,
        labels_nodes_map: typing.Optional[dict[str, "Node"]] = None,  # label -> Node
        labels_edges_map: typing.Optional[
            dict[tuple[str, str, str], "Edge"]
        ] = None,  # from_node, to_node, relation
    ) -> tuple[typing.List["Node"], typing.List["Edge"]]:
        from dvs.types.edge import Edge
        from dvs.types.node import Node

        canonical_map = {} if canonical_map is None else canonical_map
        labels_nodes_map = {} if labels_nodes_map is None else labels_nodes_map
        labels_edges_map = {} if labels_edges_map is None else labels_edges_map
        output_nodes = []
        output_edges = []

        label_norm = canonical_map.get(self.value, self.value)

        # Handle entity node
        if label_norm not in labels_nodes_map:
            entity_node = labels_nodes_map[label_norm] = Node(
                label=label_norm, kind="entity", entity=self.name
            )
        else:
            entity_node = labels_nodes_map[label_norm]
        output_nodes.append(entity_node)

        # Handle document node
        if self.document_id not in labels_nodes_map:
            doc_node = labels_nodes_map[self.document_id] = Node(
                label=self.document_id, kind="document"
            )
        else:
            doc_node = labels_nodes_map[self.document_id]
        output_nodes.append(doc_node)

        # Handle edge
        relation = "is_from"
        if (entity_node.label, doc_node.label, relation) not in labels_edges_map:
            doc_edge = labels_edges_map[
                (entity_node.label, doc_node.label, relation)
            ] = Edge(
                from_node_id=entity_node.node_id,
                from_node_label=entity_node.label,
                to_node_id=doc_node.node_id,
                to_node_label=doc_node.label,
                relation=relation,
            )
        else:
            doc_edge = labels_edges_map[(entity_node.label, doc_node.label, relation)]
        output_edges.append(doc_edge)

        return (output_nodes, output_edges)
