import typing

from ner_agent import Triplet as NerTriplet

if typing.TYPE_CHECKING:
    from dvs.types.edge import Edge
    from dvs.types.node import Node


class Triplet(NerTriplet):
    relation: typing.Literal["is_a", "has_a", "related_to", "is_from"]
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

        subject_norm = canonical_map.get(self.subject, self.subject)
        object_norm = canonical_map.get(self.object, self.object)

        # Handle document node
        if self.document_id not in labels_nodes_map:
            doc_node = labels_nodes_map[self.document_id] = Node(
                label=self.document_id, kind="document"
            )
        else:
            doc_node = labels_nodes_map[self.document_id]
        output_nodes.append(doc_node)

        # Handle subject node
        if subject_norm not in labels_nodes_map:
            subject_node = labels_nodes_map[subject_norm] = Node(
                label=subject_norm, kind="entity"
            )
        else:
            subject_node = labels_nodes_map[subject_norm]
        output_nodes.append(subject_node)

        # Handle object node
        if object_norm not in labels_nodes_map:
            object_node = labels_nodes_map[object_norm] = Node(
                label=object_norm, kind="entity"
            )
        else:
            object_node = labels_nodes_map[object_norm]
        output_nodes.append(object_node)

        # Handle subject-document edge
        if (doc_node.label, subject_norm, "is_from") not in labels_edges_map:
            _edge = labels_edges_map[(doc_node.label, subject_norm, "is_from")] = Edge(
                from_node_id=doc_node.node_id,
                from_node_label=doc_node.label,
                to_node_id=subject_node.node_id,
                to_node_label=subject_node.label,
                relation="is_from",
            )
        else:
            _edge = labels_edges_map[(doc_node.label, subject_norm, "is_from")]
        output_edges.append(_edge)

        # Handle object-document edge
        if (doc_node.label, object_norm, "is_from") not in labels_edges_map:
            _edge = labels_edges_map[(doc_node.label, object_norm, "is_from")] = Edge(
                from_node_id=doc_node.node_id,
                from_node_label=doc_node.label,
                to_node_id=object_node.node_id,
                to_node_label=object_node.label,
                relation="is_from",
            )
        else:
            _edge = labels_edges_map[(doc_node.label, object_norm, "is_from")]
        output_edges.append(_edge)

        # Handle triplet edge
        if (subject_norm, object_norm, self.relation) not in labels_edges_map:
            _edge = labels_edges_map[(subject_norm, object_norm, self.relation)] = Edge(
                from_node_id=labels_nodes_map[subject_norm].node_id,
                from_node_label=labels_nodes_map[subject_norm].label,
                to_node_id=labels_nodes_map[object_norm].node_id,
                to_node_label=labels_nodes_map[object_norm].label,
                relation=self.relation,
            )
        else:
            _edge = labels_edges_map[(subject_norm, object_norm, self.relation)]
        output_edges.append(_edge)

        return (output_nodes, output_edges)
