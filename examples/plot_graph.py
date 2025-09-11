import json
import pathlib

import networkx as nx
import plotly.graph_objects as go

data_root = pathlib.Path(__file__).parent.parent.joinpath("data")
graph_stem_name = "knowledge_graph"
graph_filepath = data_root.joinpath(f"{graph_stem_name}.json")


def main():
    graph_data = json.loads(graph_filepath.read_text())

    # Create NetworkX graph from JSON
    G = nx.DiGraph()

    # Add nodes
    for node in graph_data["nodes"]:
        G.add_node(node["id"])

    # Add edges with labels
    for link in graph_data["links"]:
        G.add_edge(link["source"], link["target"], label=link["label"])

    # Get positions for nodes
    pos = nx.spring_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.spectral_layout(G)
    # pos = nx.circular_layout(G)
    # pos = nx.shell_layout(G)

    # Create edge traces with different colors
    edge_traces = []

    # Color mapping for edge types
    color_map = {"is_a": "blue", "has_a": "green", "related_to": "red"}

    for edge_type, color in color_map.items():
        edge_x = []
        edge_y = []
        edge_text = []

        for edge in G.edges(data=True):
            if edge[2]["label"] == edge_type:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_text.append(f"{edge[0]} -> {edge[1]} ({edge_type})")

        if edge_x:  # Only add trace if there are edges of this type
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=2, color=color),
                hoverinfo="text",
                text=edge_text,
                mode="lines",
                name=edge_type,
                showlegend=True,
            )
            edge_traces.append(edge_trace)

    # Create node trace
    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            colorscale="YlGnBu",
            size=10,
            color="lightblue",
            line_width=2,
        ),
        name="Nodes",
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    # Update layout
    fig.update_layout(
        title=dict(text="Knowledge Graph Visualization", font=dict(size=16)),
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    # Save as HTML
    output_path = pathlib.Path(__file__).parent.joinpath("knowledge_graph.html")
    fig.write_html(str(output_path))
    print(f"Graph saved to {output_path}")


if __name__ == "__main__":
    main()
