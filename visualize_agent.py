from librarian.agent import librarian
from agents.extensions.visualization import draw_graph

if __name__ == "__main__":
    # Show the graph in a window
    graph = draw_graph(librarian)
    graph.view()
    # Save the graph as a PNG file
    output_file = "librarian_agent_graph.png"
    graph.render(filename=output_file, format="png", cleanup=True)
    print(f"Agent graph saved as {output_file}")



