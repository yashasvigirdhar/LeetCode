
public final class UnionFind {
  private final Component[] components;

  public UnionFind(int size) {
    components = new Component[size];
    for (int i = 0; i < size; i++) {
      components[i] = new Component(i, 1);
    }
  }

  private int findRoot(int value) {
    int root = value;
    while (root != components[root].parent) {
      root = components[root].parent;
    }

    return root;
  }

  public final boolean checkIfConnected(int v1, int v2) {
    return findRoot(v1) == findRoot(v2);
  }

  public final void union(int v1, int v2) {
    int root1 = findRoot(v1);
    int root2 = findRoot(v2);
    if (components[root1].size < components[root2].size) {
      components[root1].parent = root2;
      components[root2].size += components[root1].size;
    } else {
      components[root2].parent = root1;
      components[root1].size += components[root2].size;
    }

  }

  public final void printContents() {
    for (Component i : components) {
      System.out.print(i.parent);
    }
  }

  class Component {
    int parent;
    int size;

    public Component(int parent, int size) {
      this.parent = parent;
      this.size = size;
    }
  }
}
