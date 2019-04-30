import java.util.Arrays;
import java.util.List;
import java.util.Stack;

public class Array1 {

  public static String solve(int A, int B, int C, int D, List<Integer> E, List<Integer> F) {
    int temp = A;
    A = B;
    B = temp;
    int[][] visited = new int[A + 1][B + 1];
    for (int i = 0; i < A + 1; i++) {
      for (int j = 0; j < B + 1; j++) {
        visited[i][j] = 0;
      }
    }
    int[][] valid = new int[A + 1][B + 1];
    int[][] reachable = new int[A + 1][B + 1];
    for (int i = 0; i < A + 1; i++) {
      for (int j = 0; j < B + 1; j++) {
        reachable[i][j] = 0;
        if (checkIfTouches(i, j, C, D, E, F)) {
          valid[i][j] = 0;
        } else {
          valid[i][j] = 1;
        }
      }
    }
    for (int i = 0; i <= A; i++) {
      for (int j = 0; j <= B; j++) {
        System.out.print(valid[i][j] + " ");
      }
      System.out.println();
    }
    if (0 == A && 0 == B) {
      return "YES";
    }
    Stack<Point> st = new Stack<>();
    if (valid[0][0] == 1) {
      st.push(new Point(0, 0));
    }
    while (!st.isEmpty()) {
      Point p = st.peek();
      Point neighbour = findNextUnvisitedNeighbour(p, A, B, visited, valid);
      if (neighbour != null) {
        visited[neighbour.x][neighbour.y] = 1;
        reachable[neighbour.x][neighbour.y] = 1;
        st.push(neighbour);
      } else {
        st.pop();
      }
    }
    return (reachable[A][B] == 1) ? "YES" : "NO";
  }

  private static Point findNextUnvisitedNeighbour(Point p, int a, int b, int[][] visited, int[][] valid) {
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        if (i == 0 && j == 0) {
          continue;
        }
        if (isValid(p.x + i, p.y + j, a, b) && visited[p.x + i][p.y + j] == 0 && valid[p.x + i][p.y + j] == 1) {
          return new Point(p.x + i, p.y + j);
        }
      }
    }
    return null;
  }

  private static boolean isValid(int x, int y, int a, int b) {
    return x >= 0 && y >= 0 && x <= a && y <= b;
  }

  private static boolean checkIfTouches(int x, int y, int n, int r, List<Integer> e, List<Integer> f) {
    for (int i = 0; i < n; i++) {
      int x2 = e.get(i), y2 = f.get(i);
      double d = Math.sqrt((x2 - x) * (x2 - x) + (y2 - y) * (y2 - y));
      if (d <= r) {
        return true;
      }
    }
    return false;
  }

  static class Point {
    int x, y;

    public Point(int x, int y) {
      this.x = x;
      this.y = y;
    }
  }

}
