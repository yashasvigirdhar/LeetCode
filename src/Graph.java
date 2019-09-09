import java.util.*;

public class Graph {

  public int[] findOrderMock(int n, int[][] prerequisites) {
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }
    int[] inDegree = new int[n];
    for (int[] p : prerequisites) {
      inDegree[p[0]]++;
      graph.get(p[1]).add(p[0]);
    }
    Deque<Integer> q = new ArrayDeque<>();
    for (int i = 0; i < n; i++) {
      if (inDegree[i] == 0) {
        q.addLast(i);
      }
    }
    int[] res = new int[n];
    int covered = 0;
    while (!q.isEmpty()) {
      Integer polled = q.pollFirst();
      res[covered++] = polled;
      for (int neighbour : graph.get(polled)) {
        if (--inDegree[neighbour] == 0) {
          q.addLast(neighbour);
        }
      }
    }
    if(covered == n){
      return res;
    }
    return new int[0];
  }

  public boolean canFinishMock(int n, int[][] prerequisites) {
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }
    int[] inDegree = new int[n];
    for (int[] p : prerequisites) {
      inDegree[p[0]]++;
      graph.get(p[1]).add(p[0]);
    }
    int covered = 0;
    Deque<Integer> q = new ArrayDeque<>();
    for (int i = 0; i < n; i++) {
      if (inDegree[i] == 0) {
        q.addLast(i);
      }
    }
    while (!q.isEmpty()) {
      Integer polled = q.pollFirst();
      covered++;
      for (int neighbour : graph.get(polled)) {
        if (--inDegree[neighbour] == 0) {
          q.addLast(neighbour);
        }
      }
    }
    return covered == n;
  }

  public int minimumSemesters(int n, int[][] relations) {
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i <= n; i++) {
      graph.add(new ArrayList<>());
    }
    int[] inDegree = new int[n + 1];
    Arrays.fill(inDegree, 0);
    for (int[] r : relations) {
      graph.get(r[0]).add(r[1]);
      inDegree[r[1]]++;
    }
    int res = 0;
    Deque<Integer> q = new ArrayDeque<>();
    for (int i = 1; i <= n; i++) {
      if (inDegree[i] == 0) {
        q.addLast(i);
      }
    }
    int covered = 0;
    while (!q.isEmpty()) {
      res++;
      int size = q.size();
      while (size > 0) {
        Integer polled = q.pollFirst();
        covered++;
        for (int neighbour : graph.get(polled)) {
          inDegree[neighbour]--;
          if (inDegree[neighbour] == 0) {
            q.addLast(neighbour);
          }
        }
        size--;
      }

    }

    return (covered == n) ? res : -1;
  }

  public int minimumCost(int n, int[][] connections) {
    int[] roots = new int[n + 1];
    for (int i = 0; i <= n; i++) {
      roots[i] = i;
    }
    Arrays.sort(connections, new Comparator<int[]>() {
      @Override
      public int compare(int[] o1, int[] o2) {
        return o1[2] - o2[2];
      }
    });
    int res = 0;
    int components = n;
    for (int[] conn : connections) {
      int i = conn[0], j = conn[1], cost = conn[2];
      int root1 = findRootIII(i, roots);
      int root2 = findRootIII(j, roots);
      if (root1 != root2) {
        roots[root1] = root2;
        components--;
        res += cost;
      }
    }
    if (components > 1) {
      return -1;
    }
    return res;
  }

  private int findRootIII(int i, int[] roots) {
    while (i != roots[i]) {
      roots[i] = roots[roots[i]];
      i = roots[i];
    }
    return i;
  }

  public int[] shortestAlternatingPaths(int n, int[][] red_edges, int[][] blue_edges) {
    List<List<Integer>> red = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      red.add(new ArrayList<>());
    }
    for (int[] edge : red_edges) {
      red.get(edge[0]).add(edge[1]);
    }

    List<List<Integer>> blue = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      blue.add(new ArrayList<>());
    }
    for (int[] edge : blue_edges) {
      blue.get(edge[0]).add(edge[1]);
    }

    Deque<PathStateColor> q = new ArrayDeque<>();
    Set<Pair<Integer, Integer>> visited = new HashSet<>();

    int[] res = new int[n];
    Arrays.fill(res, -1);
    res[0] = 0;
    visited.add(new Pair<>(0, 0));
    visited.add(new Pair<>(0, 1));

    for (int neighbour : red.get(0)) {
      q.addLast(new PathStateColor(neighbour, 0, 1));
    }

    for (int neighbour : blue.get(0)) {
      q.addLast(new PathStateColor(neighbour, 1, 1));
    }

    while (!q.isEmpty()) {
      PathStateColor polled = q.pollFirst();
      if (res[polled.idx] == -1) {
        res[polled.idx] = polled.cost;
      }
      List<Integer> neighbours;
      int nextColor;
      if (polled.color == 0) {
        neighbours = blue.get(polled.idx);
        nextColor = 1;
      } else {
        neighbours = red.get(polled.idx);
        nextColor = 0;
      }
      for (int neighbour : neighbours) {
        Pair<Integer, Integer> p = new Pair<>(neighbour, nextColor);
        if (!visited.contains(p)) {
          q.addLast(new PathStateColor(neighbour, nextColor, polled.cost + 1));
          visited.add(p);
        }
      }
    }
    return res;
  }

  class PathStateColor {
    int idx;
    int color;
    int cost;

    public PathStateColor(int idx, int color, int cost) {
      this.idx = idx;
      this.color = color;
      this.cost = cost;
    }
  }

  public int minMalwareSpread(int[][] graph, int[] initial) {
    int n = graph.length;

    int[] score = new int[n];
    Arrays.fill(score, -1);

    boolean[] virus = new boolean[n];
    for (int i : initial) {
      virus[i] = true;
    }

    boolean[] visited = new boolean[n];
    Deque<Integer> q = new ArrayDeque<>();

    for (int i : initial) {
      if (score[i] != -1) {
        continue;
      }
      score[i] = 0;
      // count score

      visited[i] = true;
      q.addLast(i);
      int count = 0;
      boolean secondVirusFound = false;
      while (!q.isEmpty()) {
        Integer polled = q.pollFirst();
        if (virus[polled] && polled != i) {
          score[polled] = 0;
          secondVirusFound = true;
        }
        count++;
        for (int j = 0; j < n; j++) {
          if (j != i && graph[polled][j] == 1 && !visited[j]) {
            visited[j] = true;
            q.addLast(j);
          }
        }
      }
      if (!secondVirusFound) {
        score[i] = count;
      }
    }

    int maxScore = -1, res = -1;
    for (int i = 0; i < n; i++) {
      if (score[i] > maxScore) {
        maxScore = score[i];
        res = i;
      }
    }
    return res;
  }

  public int knightDialerDP(int N) {
    int[][] nextMoves = {
        {4, 6},
        {6, 8},
        {7, 9},
        {4, 8},
        {3, 9, 0},
        {},
        {1, 7, 0},
        {2, 6},
        {1, 3},
        {2, 4}
    };
    int[][] dp = new int[10][5001];
    for (int i = 0; i <= 9; i++) {
      dp[i][0] = 0;
      dp[i][1] = 1;
    }
    for (int i = 2; i <= N; i++) {
      for (int j = 0; j <= 9; j++) {
        dp[j][i] = 0;
        for (int next : nextMoves[j]) {
          dp[j][i] = (dp[j][i] + dp[next][i - 1]) % 1000000007;
        }
      }
    }
    int res = 0;
    for (int i = 0; i <= 9; i++) {
      res = (res + dp[i][N]) % 1000000007;
    }
    return res;
  }

  private int knightUtil(int idx, int n, int[][] dp, int[][] nextMoves) {
    if (n == 0) {
      return 1;
    }
    if (dp[idx][n] != -1) {
      return dp[idx][n];
    }
    int res = 0;
    for (int next : nextMoves[idx]) {
      res = (res + knightUtil(next, n - 1, dp, nextMoves)) % 1000000007;
    }
    dp[idx][n] = res;
    return dp[idx][n];
  }

  public int knightDialerBFS(int N) {
    int[][] nextMoves = {
        {4, 6},
        {6, 8},
        {7, 9},
        {4, 8},
        {3, 9, 0},
        {},
        {1, 7, 0},
        {2, 6},
        {1, 3},
        {2, 4}
    };
    Deque<KnightState> q = new ArrayDeque<>();
    for (int i = 0; i <= 9; i++) {
      if (i == 5 && N > 1) {
        continue;
      }
      q.addLast(new KnightState(i, N - 1));
    }
    int res = 0;
    int mod = 1000000007;
    while (!q.isEmpty()) {
      KnightState polled = q.pollFirst();
      if (polled.left == 0) {
        // increment res
        res = (res + 1) % mod;
        continue;
      }
      for (int next : nextMoves[polled.cur]) {
        q.addLast(new KnightState(next, polled.left - 1));
      }
    }
    return res;
  }

  class KnightState {
    int cur;
    int left;

    public KnightState(int cur, int left) {
      this.cur = cur;
      this.left = left;
    }
  }

  public int earliestAcq(int[][] logs, int n) {
    Arrays.sort(logs, new Comparator<int[]>() {
      @Override
      public int compare(int[] o1, int[] o2) {
        return o1[0] - o2[0];
      }
    });
    int[] roots = new int[n];
    for (int i = 0; i < n; i++) {
      roots[i] = i;
    }
    for (int[] log : logs) {
      int root1 = findRootAgain(log[1], roots);
      int root2 = findRootAgain(log[2], roots);
      if (root1 != root2) {
        n--;
        roots[root1] = root2;
        if (n == 1) {
          return log[0];
        }
      }
    }
    return -1;
  }

  private int findRootAgain(int i, int[] roots) {
    if (roots[i] != i) {
      roots[i] = findRootAgain(roots[i], roots);
    }
    return roots[i];
  }

  public int maximumMinimumPath(int[][] A) {
    int m = A.length;
    if (m == 0) {
      return 0;
    }
    int n = A[0].length;

    PriorityQueue<PathStateMan> q = new PriorityQueue<>(new Comparator<PathStateMan>() {
      @Override
      public int compare(PathStateMan o1, PathStateMan o2) {
        return o2.min - o1.min;
      }
    });


    boolean[][] visited = new boolean[m][n];
    visited[0][0] = true;

    PathStateMan e = new PathStateMan(0, 0, A[0][0]);
    q.add(e);

    while (!q.isEmpty()) {
      PathStateMan polled = q.poll();
      int curX = polled.curX;
      int curY = polled.curY;
      int curMin = polled.min;
      if (curX == m - 1 && curY == n - 1) {
        return curMin;
      }
      for (int i = 0; i < 4; i++) {
        int newX = curX + dx[i][0];
        int newY = curY + dx[i][1];
        if (isValidPoint(m, n, newX, newY) && !visited[newX][newY]) {
          int newMin = Math.min(curMin, A[newX][newY]);
          PathStateMan e1 = new PathStateMan(newX, newY, newMin);
          q.add(e1);
          visited[newX][newY] = true;
        }
      }
    }
    return -1;
  }

  class PathStateMan {
    int curX, curY;
    int min;

    public PathStateMan(int curX, int curY, int min) {
      this.curX = curX;
      this.curY = curY;
      this.min = min;
    }
  }

  public int numEnclaves(int[][] A) {
    int m = A.length;
    if (m == 0) return 0;
    int n = A[0].length;

    boolean[][] visited = new boolean[m][n];
    for (boolean[] arr : visited) {
      Arrays.fill(arr, false);
    }
    int res = 0;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (A[i][j] == 1 && !visited[i][j]) {
          int[] count = new int[]{0};
          boolean[] boundaryReached = new boolean[]{false};
          dfsEnclaves(i, j, A, visited, boundaryReached, count);
          if (!boundaryReached[0]) {
            res += count[0];
          }
        }
      }
    }
    return res;
  }

  private void dfsEnclaves(int x, int y, int[][] A, boolean[][] visited, boolean[] boundaryReached, int[] count) {
    visited[x][y] = true;
    count[0]++;
    for (int i = 0; i < 4; i++) {
      int newX = x + dx[i][0];
      int newY = y + dx[i][1];
      if (isValidPoint(A.length, A[0].length, newX, newY)) {
        if (A[newX][newY] == 1 && !visited[newX][newY]) {
          dfsEnclaves(newX, newY, A, visited, boundaryReached, count);
        }
      } else {
        boundaryReached[0] = true;
      }
    }
  }

  public boolean canVisitAllRoomsShort(List<List<Integer>> rooms) {
    int n = rooms.size();
    Set<Integer> visited = new HashSet<>();
    visited.add(0);
    Deque<Integer> q = new ArrayDeque<>();
    q.addLast(0);
    while (!q.isEmpty()) {
      Integer polled = q.pollFirst();
      for (int neighbour : rooms.get(polled)) {
        if (!visited.contains(neighbour)) {
          visited.add(neighbour);
          q.addLast(neighbour);
          if (visited.size() == n) {
            return true;
          }
        }
      }
    }
    return false;
  }

  public boolean canVisitAllRooms(List<List<Integer>> rooms) {
    int n = rooms.size();

    boolean[] unlocked = new boolean[n];
    Arrays.fill(unlocked, false);

    Deque<PathStat> q = new ArrayDeque<>();
    boolean[] visited = new boolean[n];
    Arrays.fill(visited, false);
    visited[0] = unlocked[0] = true;
    q.addLast(new PathStat(0, visited));

    Set<PathStat> set = new HashSet<>();
    set.add(new PathStat(0, visited));


    while (!q.isEmpty()) {
      PathStat curState = q.pollFirst();
      boolean[] curVisited = curState.visited;
      if (checkIfAllVisited(unlocked)) {
        return true;
      }
      int cur = curState.node;
      for (int neighbour : rooms.get(cur)) {
        boolean[] newVisited = new boolean[n];
        System.arraycopy(curVisited, 0, newVisited, 0, n);
        newVisited[neighbour] = true;
        unlocked[neighbour] = true;
        PathStat ps = new PathStat(neighbour, newVisited);
        if (!set.contains(ps)) {
          set.add(ps);
          q.addLast(ps);
        }
      }
    }
    return false;
  }

  class PathStat {
    int node;
    boolean[] visited;

    public PathStat(int node, boolean[] visited) {
      this.node = node;
      this.visited = visited;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      PathStat pathStat = (PathStat) o;
      return node == pathStat.node &&
          Arrays.equals(visited, pathStat.visited);
    }

    @Override
    public int hashCode() {
      int result = Objects.hash(node);
      result = 31 * result + Arrays.hashCode(visited);
      return result;
    }
  }


  private boolean checkIfAllVisited(boolean[] visited) {
    for (boolean b : visited) {
      if (!b) {
        return false;
      }
    }
    return true;
  }

  public int shortestPathLength(int[][] graph) {

    int n = graph.length;
    Deque<PathState> q = new ArrayDeque<>();
    Set<PathState> set = new HashSet<>();
    for (int i = 0; i < n; i++) {
      int mask = 1 << i;
      q.addLast(new PathState(mask, 0, i));
      set.add(new PathState(mask, 0, i));
    }
    while (!q.isEmpty()) {
      PathState polled = q.pollFirst();
      if (polled.visitedMask == ((1 << n + 1) - 1)) {
        return polled.cost;
      }

      for (int neighbour : graph[polled.curr]) {
        int newMask = polled.visitedMask | 1 << neighbour;
        PathState state = new PathState(newMask, polled.cost + 1, neighbour);
        if (!set.contains(state)) {
          q.addLast(state);
          set.add(state);
        }
      }
    }
    return 0;
  }

  class PathState {
    int visitedMask;
    int cost;
    int curr;

    public PathState(int visitedMask, int cost, int curr) {
      this.visitedMask = visitedMask;
      this.cost = cost;
      this.curr = curr;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      PathState pathState = (PathState) o;
      return visitedMask == pathState.visitedMask &&
          cost == pathState.cost &&
          curr == pathState.curr;
    }

    @Override
    public int hashCode() {
      return Objects.hash(visitedMask, cost, curr);
    }
  }

  public List<Integer> eventualSafeNodes(int[][] graph) {
    int n = graph.length;
    int[] safe = new int[n];
    Arrays.fill(safe, -1);
    for (int i = 0; i < n; i++) {
      if (safe[i] == -1) {
        eventualSafeNodesUtil(i, graph, safe);
      }
    }
    List<Integer> res = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      if (safe[i] == 1) {
        res.add(i);
      }
    }
    return res;
  }

  private boolean eventualSafeNodesUtil(int idx, int[][] graph, int[] safe) {
    if (safe[idx] != -1) {
      return safe[idx] == 1;
    }
    safe[idx] = 0;
    for (int neighbour : graph[idx]) {
      if (!eventualSafeNodesUtil(neighbour, graph, safe)) {
        safe[idx] = 0;
        return false;
      }

    }
    safe[idx] = 1;
    return true;
  }

  public boolean possibleBipartitionUsingDFS(int n, int[][] dislikes) {
    List<List<Integer>> g = new ArrayList<>();
    for (int i = 0; i <= n; i++) {
      g.add(new ArrayList<>());
    }
    for (int[] d : dislikes) {
      g.get(d[0]).add(d[1]);
      g.get(d[1]).add(d[0]);
    }

    int[] color = new int[n + 1];
    Arrays.fill(color, 0);
    for (int i = 1; i <= n; i++) {
      if (color[i] != 0) {
        continue;
      }
      color[i] = 1;
      Deque<Integer> stack = new ArrayDeque<>();
      stack.addFirst(i);
      while (!stack.isEmpty()) {
        Integer peek = stack.peekFirst();
        boolean unvisitedFound = false;
        for (int neighbour : g.get(peek)) {
          if (color[neighbour] == 0) {
            unvisitedFound = true;
            color[neighbour] = -1 * color[peek];
            stack.addFirst(neighbour);
            break;
          }
          if (color[neighbour] == color[peek]) {
            return false;
          }
        }
        if (!unvisitedFound) {
          stack.pollFirst();
        }
      }
    }
    return true;
  }

  public boolean possibleBipartitionUsingBFS(int n, int[][] dislikes) {
    List<List<Integer>> g = new ArrayList<>();
    for (int i = 0; i <= n; i++) {
      g.add(new ArrayList<>());
    }
    for (int[] d : dislikes) {
      g.get(d[0]).add(d[1]);
      g.get(d[1]).add(d[0]);
    }

    int[] color = new int[n + 1];
    Arrays.fill(color, 0);
    for (int i = 1; i <= n; i++) {
      if (color[i] != 0) {
        continue;
      }
      Deque<Integer> q = new ArrayDeque<>();
      color[i] = 1;
      q.addLast(i);
      while (!q.isEmpty()) {
        Integer polled = q.pollFirst();
        for (int neighbour : g.get(polled)) {
          if (color[neighbour] == color[polled]) {
            return false;
          }
          if (color[neighbour] == 0) {
            if (!assignColor(color, polled, neighbour, g)) {
              return false;
            }
            q.addLast(neighbour);
          }
        }
      }
    }
    return true;
  }

  private boolean assignColor(int[] color, Integer polled, int neighbour, List<List<Integer>> g) {
    Set<Integer> distinctColors = new HashSet<>();
    for (int n : g.get(neighbour)) {
      if (color[n] != 0) {
        distinctColors.add(color[n]);
      }
    }
    if (distinctColors.size() < 2) {
      color[neighbour] = -1 * color[polled];
      return true;
    }
    return false;
  }

  public int numberOfPatterns(int m, int n) {
    int[][] skip = new int[10][10];
    skip[1][3] = skip[3][1] = 2;
    skip[1][7] = skip[7][1] = 4;
    skip[3][9] = skip[9][3] = 6;
    skip[7][9] = skip[9][7] = 8;
    skip[1][9] = skip[9][1] = skip[2][8] = skip[8][2] = skip[3][7] = skip[7][3] = skip[4][6] = skip[6][4] = 5;

    Deque<LockNode> q = new ArrayDeque<>();
    for (int i = 1; i <= 9; i++) {
      q.addLast(new LockNode(i, 1, new HashSet<>(Collections.singletonList(i))));
    }
    int res = 0;
    while (!q.isEmpty()) {
      LockNode polled = q.pollFirst();
      if (polled.pathLength >= m) {
        res++;
      }
      if (polled.pathLength == n) {
        continue;
      }
      for (int i = 1; i <= 9; i++) {
        if (!polled.visited.contains(i)
            && (skip[polled.val][i] == 0 || polled.visited.contains(skip[polled.val][i]))) {
          Set<Integer> newVisited = new HashSet<>(polled.visited);
          newVisited.add(i);
          q.addLast(new LockNode(i, polled.pathLength + 1, newVisited));
        }

      }
    }
    return res;
  }

  class LockNode {
    int val;
    int pathLength;
    Set<Integer> visited;

    public LockNode(int val, int pathLength, Set<Integer> visited) {
      this.val = val;
      this.pathLength = pathLength;
      this.visited = visited;
    }
  }

  public int[] gardenNoAdj(int n, int[][] paths) {
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i <= n; i++) {
      graph.add(new ArrayList<>());
    }
    for (int[] path : paths) {
      graph.get(path[0]).add(path[1]);
      graph.get(path[1]).add(path[0]);
    }
    int[] colors = new int[n + 1];
    Arrays.fill(colors, -1);
    for (int i = 1; i <= n; i++) {
      if (colors[i] != -1) {
        continue;
      }
      colors[i] = 1;
      Stack<Integer> st = new Stack<>();
      st.push(i);
      while (!st.empty()) {
        Integer peek = st.peek();
        boolean unvisitedFound = false;
        for (int neighbour : graph.get(peek)) {
          if (colors[neighbour] == -1) {
            assignColor(neighbour, graph, colors);
            st.push(neighbour);
            unvisitedFound = true;
            break;
          }
        }
        if (!unvisitedFound) {
          st.pop();
        }
      }
    }
    int[] res = new int[n];
    if (n >= 0) System.arraycopy(colors, 1, res, 0, n);
    return res;
  }

  private void assignColor(int idx, List<List<Integer>> graph, int[] colors) {
    boolean[] colorsLeft = new boolean[5];
    Arrays.fill(colorsLeft, true);
    for (int neighbour : graph.get(idx)) {
      if (colors[neighbour] != -1) {
        colorsLeft[colors[neighbour]] = false;
      }
    }
    for (int i = 1; i <= 4; i++) {
      if (colorsLeft[i]) {
        colors[idx] = i;
        return;
      }
    }
  }

  public int[] findRedundantDirectedConnection(int[][] edges) {
    int n = edges.length;
    int[] parent = new int[n + 1];
    Arrays.fill(parent, -1);
    int[][] candidateEdges = new int[][]{{-1, -1}, {-1, -1}};
    int targetVertex = -1;

    for (int[] edge : edges) {
      if (parent[edge[1]] != -1) {
        candidateEdges[0] = new int[]{parent[edge[1]], edge[1]};
        candidateEdges[1] = edge;
        targetVertex = edge[1];
        break;
      }
      parent[edge[1]] = edge[0];
    }


    // check if loop present
    int[] root = new int[n + 1];
    for (int i = 1; i <= n; i++) {
      root[i] = i;
    }
    for (int[] edge : edges) {
      int u = edge[0];
      int v = edge[1];
      if (targetVertex != 1 && (candidateEdges[1][0] == u && candidateEdges[1][1] == v)) {
        continue;
      }
      int parentU = findRootII(u, root);
      // if parent of u is already v
      if (parentU == v) {
        // loop found
        if (targetVertex == -1) {
          // no candidate edges for vertex with in-degree two. return this edge
          return edge;
        } else {
          // candidate edges exist, one of them will match with this one
          return candidateEdges[0];
        }
      } else {
        root[v] = parentU;
      }
    }

    // no loop, candidate Edges defintely exist
    return candidateEdges[1];
  }

  public double[] calcEquationUsingDFS(List<List<String>> equations, double[] values, List<List<String>> queries) {
    Map<String, Map<String, Double>> graph = new HashMap<>();
    for (int i = 0; i < equations.size(); i++) {
      List<String> equation = equations.get(i);
      String num = equation.get(0);
      String deno = equation.get(1);

      double val = values[i];
      if (!graph.containsKey(num)) {
        graph.put(num, new HashMap<>());
      }
      graph.get(num).put(deno, val);

      double inverseVal = 1 / values[i];
      if (!graph.containsKey(deno)) {
        graph.put(deno, new HashMap<>());
      }
      graph.get(deno).put(num, inverseVal);
    }

    double[] res = new double[queries.size()];
    for (int i = 0; i < queries.size(); i++) {
      List<String> query = queries.get(i);
      String num = query.get(0);
      String deno = query.get(1);
      if (!graph.containsKey(num)) {
        res[i] = -1;
        continue;
      }
      if (num.equals(deno)) {
        res[i] = 1.0;
        continue;
      }
      Map<String, Boolean> visited = new HashMap<>();
      visited.put(num, true);
      Stack<Pair<String, Double>> st = new Stack<>();
      st.push(new Pair<>(num, 1.0));
      boolean denoFound = false;
      while (!st.empty()) {
        Pair<String, Double> polled = st.peek();
        Double ansTillNow = polled.second;
        Map<String, Double> possibleDenos = graph.get(polled.first);
        boolean unvisitedNeighbourFound = false;
        for (Map.Entry<String, Double> entry : possibleDenos.entrySet()) {
          if (visited.containsKey(entry.getKey())) {
            continue;
          }
          unvisitedNeighbourFound = true;
          double newAns = entry.getValue() * ansTillNow;
          if (entry.getKey().equals(deno)) {
            res[i] = newAns;
            denoFound = true;
          } else {
            visited.put(entry.getKey(), true);
            st.push(new Pair<>(entry.getKey(), newAns));
          }
          break;
        }
        if (denoFound) {
          break;
        }
        if (!unvisitedNeighbourFound) {
          st.pop();
        }
      }
      if (!denoFound) {
        res[i] = -1;
      }
    }
    return res;
  }

  public double[] calcEquationUsingBFS(List<List<String>> equations, double[] values, List<List<String>> queries) {
    Map<String, Map<String, Double>> graph = new HashMap<>();
    for (int i = 0; i < equations.size(); i++) {
      List<String> equation = equations.get(i);
      String num = equation.get(0);
      String deno = equation.get(1);

      double val = values[i];
      if (!graph.containsKey(num)) {
        graph.put(num, new HashMap<>());
      }
      graph.get(num).put(deno, val);

      double inverseVal = 1 / values[i];
      if (!graph.containsKey(deno)) {
        graph.put(deno, new HashMap<>());
      }
      graph.get(deno).put(num, inverseVal);
    }

    double[] res = new double[queries.size()];
    for (int i = 0; i < queries.size(); i++) {
      List<String> query = queries.get(i);
      String num = query.get(0);
      String deno = query.get(1);
      if (!graph.containsKey(num)) {
        res[i] = -1;
        continue;
      }
      if (num.equals(deno)) {
        res[i] = 1.0;
      }
      Map<String, Boolean> visited = new HashMap<>();
      Queue<Pair<String, Double>> queue = new LinkedList<>();
      queue.add(new Pair<>(num, 1.0));
      visited.put(num, true);
      boolean denoFound = false;
      while (!queue.isEmpty()) {
        Pair<String, Double> polled = queue.poll();
        Double ansTillNow = polled.second;
        Map<String, Double> possibleDenos = graph.get(polled.first);
        for (Map.Entry<String, Double> entry : possibleDenos.entrySet()) {
          if (visited.containsKey(entry.getKey())) {
            continue;
          }
          if (entry.getKey().equals(deno)) {
            res[i] = entry.getValue() * ansTillNow;
            denoFound = true;
            break;
          }
          visited.put(entry.getKey(), true);
          queue.add(new Pair<>(entry.getKey(), entry.getValue() * ansTillNow));
        }
        if (denoFound) {
          break;
        }
      }
      if (!denoFound) {
        res[i] = -1;
      }
    }
    return res;
  }

  public int findCelebrity(int n) {
    int[] degree = new int[n];
    Arrays.fill(degree, 0);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (j == i) {
          continue;
        }
        if (knows(i, j)) {
          degree[j]++;
          degree[i]--;
        }
      }
    }

    int celebrity = -1;
    for (int i = 0; i < n; i++) {
      if (degree[i] == n - 1) {
        if (celebrity == -1) {
          celebrity = i;
        } else {
          return -1;
        }
      }
    }
    return celebrity;
  }


  boolean knows(int a, int b) {
    return false;
  }

  public int findJudge(int n, int[][] trust) {
    int[] degree = new int[n + 1];
    Arrays.fill(degree, 0);
    for (int[] t : trust) {
      degree[t[1]]++;
      degree[t[0]]--;
    }

    int judge = -1;
    for (int i = 1; i <= n; i++) {
      if (degree[i] == n - 1) {
        if (judge == -1) {
          judge = i;
        } else {
          return -1;
        }
      }
    }
    return judge;
  }

  public int[] findRedundantConnection(int[][] edges) {
    int n = edges.length;
    int[] roots = new int[n + 1];
    for (int i = 1; i <= n; i++) {
      roots[i] = i;
    }
    for (int[] edge : edges) {
      int root1 = findRootII(edge[0], roots);
      int root2 = findRootII(edge[1], roots);
      if (root1 == root2) {
        return edge;
      } else {
        roots[root1] = root2;
      }
    }
    return new int[0];
  }

  private int findRootII(int i, int[] roots) {
    while (roots[i] != i) {
      roots[i] = roots[roots[i]];
      i = roots[i];
    }
    return i;
  }

  private int[][] dx = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

  public String findShortestWay(int[][] maze, int[] start, int[] destination) {
    int m = maze.length;
    int n = maze[0].length;

    int[][] distance = new int[m][n];
    for (int[] arr : distance) {
      Arrays.fill(arr, Integer.MAX_VALUE);
    }
    distance[start[0]][start[1]] = 0;

    String[][] path = new String[m][n];
    path[start[0]][start[1]] = "";

    Queue<int[]> queue = new LinkedList<>();
    queue.add(new int[]{start[0], start[1]});
    while (!queue.isEmpty()) {
      int[] polled = queue.poll();
      int x = polled[0], y = polled[1];
      if (isDestination(destination, x, y)) {
        continue;
      }
      for (int i = 0; i < 4; i++) {
        StringBuilder pathTillNow = new StringBuilder(path[x][y]);
        int newX = x + dx[i][0];
        int newY = y + dx[i][1];
        int d = 1;
        if (!isValidPoint(m, n, newX, newY) || maze[newX][newY] == 1) {
          continue;
        }
        switch (i) {
          case 0:
            pathTillNow.append("u");
            while (!isDestination(destination, newX, newY) && isValidPoint(m, n, newX - 1, newY) && maze[newX - 1][newY] == 0) {
              newX--;
              d++;
            }
            break;
          case 1:
            pathTillNow.append("r");
            while (!isDestination(destination, newX, newY) && isValidPoint(m, n, newX, newY + 1) && maze[newX][newY + 1] == 0) {
              newY++;
              d++;

            }
            break;
          case 2:
            pathTillNow.append("d");
            while (!isDestination(destination, newX, newY) && isValidPoint(m, n, newX + 1, newY) && maze[newX + 1][newY] == 0) {
              newX++;
              d++;
            }
            break;
          case 3:
            pathTillNow.append("l");
            while (!isDestination(destination, newX, newY) && isValidPoint(m, n, newX, newY - 1) && maze[newX][newY - 1] == 0) {
              newY--;
              d++;
            }
            break;
        }
        if ((distance[x][y] + d < distance[newX][newY]) ||
            ((distance[x][y] + d == distance[newX][newY] && pathTillNow.toString().compareTo(path[newX][newY]) < 0))) {
          distance[newX][newY] = distance[x][y] + d;
          path[newX][newY] = pathTillNow.toString();
          queue.add(new int[]{newX, newY});
        }
      }
    }
    return distance[destination[0]][destination[1]] == Integer.MAX_VALUE ? "impossible" : path[destination[0]][destination[1]];
  }

  private boolean isDestination(int[] destination, int newX, int newY) {
    return newX == destination[0] && newY == destination[1];
  }

  public int shortestDistance(int[][] maze, int[] start, int[] destination) {
    int m = maze.length;
    int n = maze[0].length;
    int[][] distance = new int[m][n];
    for (int[] arr : distance) {
      Arrays.fill(arr, Integer.MAX_VALUE);
    }
    distance[start[0]][start[1]] = 0;
    Queue<int[]> queue = new LinkedList<>();
    queue.add(new int[]{start[0], start[1]});
    int minDist = Integer.MAX_VALUE;
    while (!queue.isEmpty()) {
      int[] polled = queue.poll();
      int x = polled[0], y = polled[1];
      if (x == destination[0] && y == destination[1]) {
        minDist = Math.min(minDist, distance[x][y]);
        continue;
      }
      for (int i = 0; i < 4; i++) {
        int newX = x + dx[i][0];
        int newY = y + dx[i][1];
        int d = 1;
        if (!isValidPoint(m, n, newX, newY) || maze[newX][newY] == 1) {
          continue;
        }
        switch (i) {
          case 0:
            while (isValidPoint(m, n, newX - 1, newY) && maze[newX - 1][newY] == 0) {
              newX--;
              d++;
            }
            break;
          case 1:
            while (isValidPoint(m, n, newX, newY + 1) && maze[newX][newY + 1] == 0) {
              newY++;
              d++;
            }
            break;
          case 2:
            while (isValidPoint(m, n, newX + 1, newY) && maze[newX + 1][newY] == 0) {
              newX++;
              d++;
            }
            break;
          case 3:
            while (isValidPoint(m, n, newX, newY - 1) && maze[newX][newY - 1] == 0) {
              newY--;
              d++;
            }
            break;
        }
        if (distance[x][y] + d < distance[newX][newY]) {
          distance[newX][newY] = distance[x][y] + d;
          queue.add(new int[]{newX, newY});
        }
      }
    }
    return minDist == Integer.MAX_VALUE ? -1 : minDist;
  }

  public boolean hasPath(int[][] maze, int[] start, int[] destination) {
    int m = maze.length;
    int n = maze[0].length;
    int[][] distance = new int[m][n];
    for (int[] arr : distance) {
      Arrays.fill(arr, Integer.MAX_VALUE);
    }
    distance[start[0]][start[1]] = 0;
    Queue<int[]> queue = new LinkedList<>();
    queue.add(new int[]{start[0], start[1]});
    while (!queue.isEmpty()) {
      int[] polled = queue.poll();
      int x = polled[0], y = polled[1];
      if (x == destination[0] && y == destination[1]) {
        return true;
      }
      for (int i = 0; i < 4; i++) {
        int newX = x + dx[i][0];
        int newY = y + dx[i][1];
        if (!isValidPoint(m, n, newX, newY) || maze[newX][newY] == 1) {
          continue;
        }
        switch (i) {
          case 0:
            while (isValidPoint(m, n, newX - 1, newY) && maze[newX - 1][newY] == 0) {
              newX--;
            }
            break;
          case 1:
            while (isValidPoint(m, n, newX, newY + 1) && maze[newX][newY + 1] == 0) {
              newY++;
            }
            break;
          case 2:
            while (isValidPoint(m, n, newX + 1, newY) && maze[newX + 1][newY] == 0) {
              newX++;
            }
            break;
          case 3:
            while (isValidPoint(m, n, newX, newY - 1) && maze[newX][newY - 1] == 0) {
              newY--;
            }
            break;
        }
        if (distance[x][y] + 1 < distance[newX][newY]) {
          distance[newX][newY] = distance[x][y] + 1;
          queue.add(new int[]{newX, newY});
        }
      }
    }
    return false;
  }

  public boolean isEscapePossible(int[][] blocked, int[] source, int[] target) {
    return isFree(blocked, source, target) && isFree(blocked, target, source);

  }

  private boolean isFree(int[][] blocked, int[] cell, int[] target) {
    Queue<int[]> queue = new LinkedList<>();
    queue.add(new int[]{cell[0], cell[1], 0});
    Set<String> visited = new HashSet<>();
    visited.add(String.valueOf(cell[0]) + '#' + cell[1]);
    while (!queue.isEmpty()) {
      int[] polled = queue.poll();
      int x = polled[0], y = polled[1];
      int d = polled[2];
      if (d == 100) {
        return true;
      }
      if (x == target[0] && y == target[1]) {
        return true;
      }
      for (int i = 0; i < 4; i++) {
        int newX = x + dx[i][0];
        int newY = y + dx[i][1];
        if (!isValidPoint(1000000, 1000000, newX, newY)) {
          continue;
        }
        String key = String.valueOf(newX) + '#' + newY;
        if (visited.contains(key)) {
          continue;
        }
        boolean isBlocked = false;
        for (int j = 0; j < blocked.length; j++) {
          if (blocked[j][0] == newX && blocked[j][1] == newY) {
            isBlocked = true;
            break;
          }
        }
        if (!isBlocked) {
          queue.add(new int[]{newX, newY, d + 1});
        }
      }
    }
    return false;
  }

  public int[][] colorBorder(int[][] grid, int r0, int c0, int color) {
    int m = grid.length;
    if (m == 0) return grid;
    int n = grid[0].length;
    int[][] res = new int[m][n];
    for (int i = 0; i < m; i++) {
      System.arraycopy(grid[i], 0, res[i], 0, n);
    }

    boolean[][] visited = new boolean[m][n];
    Queue<int[]> queue = new LinkedList<>();
    queue.add(new int[]{r0, c0});
    visited[r0][c0] = true;
    while (!queue.isEmpty()) {
      int[] polled = queue.poll();
      int x = polled[0], y = polled[1];
      boolean shouldColor = false;
      for (int i = 0; i < 4; i++) {
        int newX = x + dx[i][0];
        int newY = y + dx[i][1];
        if (!isValidPoint(m, n, newX, newY) || grid[newX][newY] != grid[x][y]) {
          shouldColor = true;
          continue;
        }
        if (!visited[newX][newY]) {
          visited[newX][newY] = true;
          queue.add(new int[]{newX, newY});
        }
      }
      if (shouldColor) {
        res[x][y] = color;
      }
    }
    return res;
  }


  public int scheduleCourseDP(int[][] courses) {
    Arrays.sort(courses, Comparator.comparingInt(o -> o[1]));
    int n = courses.length;
    int maxEnd = courses[n - 1][1];
    int[] time = new int[maxEnd + 2];
    Arrays.fill(time, 0);
    for (int i = n - 1; i >= 0; i--) {
      int duration = courses[i][0];
      int closeTime = courses[i][1];
      int maxStart = closeTime - duration + 1;
      for (int j = 1; j <= maxStart; j++) {
        time[j] = Math.max(time[j], 1 + time[j + duration]);
      }
    }
    return time[1];
  }

  public int scheduleCourse(int[][] courses) {
    int n = courses.length;
    boolean[] chosen = new boolean[n];
    Arrays.fill(chosen, false);
    return scheduleCourseUtil(1, courses, chosen);
  }

  private int scheduleCourseUtil(int timestamp, int[][] courses, boolean[] chosen) {
    int ans = 0;
    for (int i = 0; i < courses.length; i++) {
      if (!chosen[i] && timestamp + courses[i][0] <= courses[i][1]) {
        chosen[i] = true;
        ans = Math.max(ans, 1 + scheduleCourseUtil(timestamp + courses[i][0], courses, chosen));
        chosen[i] = false;
      }
    }
    return ans;
  }

  public int minSwapsCouples(int[] row) {
    int n = row.length / 2;
    int[][] graph = new int[n][n];
    for (int[] arr : graph) {
      Arrays.fill(arr, 0);
    }

    for (int i = 0; i < row.length - 1; i += 2) {
      int curA = row[i], curB = row[i + 1];

      int findA = (curA % 2 == 0) ? curA + 1 : curA - 1;
      int findB = (curB % 2 == 0) ? curB + 1 : curB - 1;
      if (curB == findA && curA == findB) {
        //already a couple
        continue;
      }
      for (int j = 0; j < row.length - 1; j += 2) {
        if (row[j] == findA || row[j + 1] == findA || row[j] == findB || row[j + 1] == findB) {
          graph[i / 2][j / 2] = 1;
          graph[j / 2][i / 2] = 1;
        }
      }
    }
    //graph formed

    boolean[] visited = new boolean[n];
    Arrays.fill(visited, false);
    int ans = 0;
    for (int i = 0; i < n; i++) {
      if (visited[i]) {
        continue;
      }
      List<Integer> neighbours = new ArrayList<>();
      for (int j = 0; j < n; j++) {
        if (graph[i][j] == 1) {
          neighbours.add(j);
        }
      }
      if (neighbours.size() == 1) {
        Integer neighbour = neighbours.get(0);
        visited[i] = true;
        visited[neighbour] = true;
        ans++;
      } else if (neighbours.size() == 2) {
        // count no of nodes in this cycle
        int count = 1;
        int next = neighbours.get(0);
        visited[i] = true;
        while (true) {
          visited[next] = true;
          count += 1;
          boolean nextFound = false;
          for (int j = 0; j < n; j++) {
            if (graph[next][j] == 1 && !visited[j]) {
              nextFound = true;
              next = j;
              break;
            }
          }
          if (!nextFound) {
            break;
          }
        }
        ans += (count - 1);
      }
    }
    return ans;
  }

  public boolean isBipartite(int[][] graph) {
    int n = graph.length;
    int[] color = new int[n];
    Arrays.fill(color, -1);
    for (int i = 0; i < n; i++) {
      if (color[i] != -1) {
        continue;
      }
      color[i] = 0;
      Queue<Integer> queue = new LinkedList<>();
      queue.add(i);
      while (!queue.isEmpty()) {
        Integer poll = queue.poll();
        for (int neighbour : graph[poll]) {
          if (color[neighbour] == -1) {
            color[neighbour] = color[poll] ^ 1;
            queue.add(neighbour);
          } else if (color[neighbour] == color[poll]) {
            return false;
          }
        }
      }
    }
    return true;
  }

  public String alienOrder(String[] words) {
    int[][] graph = new int[26][26];
    for (int[] g : graph) {
      Arrays.fill(g, 0);
    }
    boolean[] isPresent = new boolean[26];
    Arrays.fill(isPresent, false);
    int uniqueChars = 0;
    for (String s : words) {
      for (int i = 0; i < s.length(); i++) {
        if (!isPresent[s.charAt(i) - 'a']) {
          isPresent[s.charAt(i) - 'a'] = true;
          uniqueChars++;
        }
      }
    }
    int[] inDegree = new int[26];
    Arrays.fill(inDegree, 0);
    for (int i = 0; i < words.length; i++) {
      for (int j = i + 1; j < words.length; j++) {
        int idx = 0, jdx = 0;
        while (idx < words[i].length() && jdx < words[j].length()) {
          if (words[i].charAt(idx) != words[j].charAt(jdx)) {
            inDegree[words[j].charAt(jdx) - 'a']++;
            graph[words[i].charAt(idx) - 'a'][words[j].charAt(jdx) - 'a'] = 1;
            break;
          }
          idx++;
          jdx++;
        }
        if (idx < words[i].length() && jdx == words[j].length()) {
          return "";
        }
      }
    }
    Queue<Integer> queue = new LinkedList<>();
    StringBuilder ans = new StringBuilder();
    for (int i = 0; i < 26; i++) {
      if (isPresent[i] && inDegree[i] == 0) {
        ans.append((char) (i + 'a'));
        queue.add(i);
      }
    }
    while (!queue.isEmpty()) {
      Integer polled = queue.poll();
      for (int i = 0; i < 26; i++) {
        if (graph[polled][i] == 1) {
          inDegree[i]--;
          if (inDegree[i] == 0) {
            ans.append((char) (i + 'a'));
            queue.add(i);
          }
        }
      }
    }
    if (ans.length() == uniqueChars) {
      return ans.toString();
    } else {
      return "";
    }
  }

  public int networkDelayTime(int[][] times, int n, int k) {

    List<List<int[]>> graph = new ArrayList<>();
    for (int i = 0; i <= n; i++) {
      graph.add(new ArrayList<>());
    }

    for (int[] time : times) {
      graph.get(time[0]).add(new int[]{time[1], time[2]});
    }
    int[] distances = new int[n + 1];
    Arrays.fill(distances, Integer.MAX_VALUE);
    distances[k] = 0;
    Queue<Integer> queue = new LinkedList<>();
    queue.add(k);
    while (!queue.isEmpty()) {
      Integer polled = queue.poll();
      List<int[]> neighbours = graph.get(polled);
      for (int[] neighbour : neighbours) {
        if (distances[polled] + neighbour[1] < distances[neighbour[0]]) {
          distances[neighbour[0]] = distances[polled] + neighbour[1];
          queue.add(neighbour[0]);
        }
      }
    }
    int ans = -1;
    for (int i = 1; i <= n; i++) {
      if (distances[i] == Integer.MAX_VALUE) {
        return -1;
      }
      ans = Math.max(ans, distances[i]);
    }
    return ans;
  }

  public int snakesAndLadders(int[][] b) {
    int n = b.length;
    if (n == 0) {
      return 0;
    }
    int[][] board = new int[n][n];
    for (int i = 0; i < n; i++) {
      System.arraycopy(b[n - 1 - i], 0, board[i], 0, n);
    }
    int[] minSteps = new int[n * n + 1];
    Arrays.fill(minSteps, Integer.MAX_VALUE);
    minSteps[1] = 0;
    Queue<Integer> queue = new LinkedList<>();
    queue.add(1);
    while (!queue.isEmpty()) {
      int k = queue.poll();
      if (k == n * n) {
        continue;
      }
      int[] curPoint = getPointFromNo(k, n);
      for (int i = 1; i <= 6 && k + i <= n * n; i++) {
        int jumpTo = k + i;
        int[] newPoint = getPointFromNo(jumpTo, n);
        int newX = newPoint[0], newY = newPoint[1];

        if (board[newX][newY] == -1) {
          if (minSteps[k] + 1 < minSteps[jumpTo]) {
            minSteps[jumpTo] = minSteps[k] + 1;
            queue.add(jumpTo);
          }
        } else if (board[newX][newY] != -1) {
          jumpTo = board[newX][newY];
          int[] jumpPoint = getPointFromNo(jumpTo, n);
          if (minSteps[k] + 1 < minSteps[jumpTo]) {
            minSteps[jumpTo] = minSteps[k] + 1;
            queue.add(jumpTo);
          }
        }
      }
    }
    return minSteps[n * n] == Integer.MAX_VALUE ? -1 : minSteps[n * n];
  }

  private int[] getPointFromNo(int k, int n) {
    int row = k / n;
    if (row % 2 != 0) {
      if (k % n == 0) {
        return new int[]{row - 1, n - 1};
      } else {
        return new int[]{row, n - k % n};
      }
    } else {
      if (k % n == 0) {
        return new int[]{row - 1, 0};
      } else {
        return new int[]{row, k % n - 1};
      }
    }
  }

  public void wallsAndGates(int[][] rooms) {
    int m = rooms.length;
    if (m == 0) {
      return;
    }
    int n = rooms[0].length;
    Queue<int[]> q = new LinkedList<>();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (rooms[i][j] == 0) {
          q.add(new int[]{i, j});
        }
      }
    }
    while (!q.isEmpty()) {
      int[] polled = q.poll();
      int x = polled[0];
      int y = polled[1];
      for (int i = 0; i < 4; i++) {
        int nx = x + dx[i][0];
        int ny = y + dx[i][1];
        if (isValidPoint(m, n, nx, ny) && rooms[nx][ny] != -1 && rooms[x][y] + 1 < rooms[nx][ny]) {
          rooms[nx][ny] = rooms[x][y] + 1;
          q.add(new int[]{nx, ny});
        }
      }
    }
  }

  public int[][] updateMatrix(int[][] matrix) {
    int m = matrix.length;
    if (m == 0) {
      return matrix;
    }
    int n = matrix[0].length;
    int[][] res = new int[m][n];
    for (int[] arr : res) {
      Arrays.fill(arr, Integer.MAX_VALUE);
    }
    Queue<int[]> q = new LinkedList<>();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (matrix[i][j] == 0) {
          res[i][j] = 0;
          q.add(new int[]{i, j});
        }
      }
    }

    while (!q.isEmpty()) {
      int[] polled = q.poll();
      int x = polled[0];
      int y = polled[1];
      for (int idx = 0; idx < 4; idx++) {
        int nx = x + dx[idx][0];
        int ny = y + dx[idx][1];
        if (isValidPoint(m, n, nx, ny) && matrix[nx][ny] == 1 && res[x][y] + 1 < res[nx][ny]) {
          res[nx][ny] = res[x][y] + 1;
          q.add(new int[]{nx, ny});
        }
      }
    }

    return res;
  }

  public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
    List<ArrayList<int[]>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }
    for (int[] flight : flights) {
      graph.get(flight[0]).add(new int[]{flight[1], flight[2]});
    }
    int[] cost = new int[n];
    Arrays.fill(cost, Integer.MAX_VALUE);
    Queue<int[]> queue = new LinkedList<>();
    queue.add(new int[]{src, 0, 1});
    cost[src] = 0;
    int minCost = Integer.MAX_VALUE;
    while (!queue.isEmpty()) {
      int[] polled = queue.poll();
      int curCity = polled[0];
      int costTillNow = polled[1];
      int citiesTillNow = polled[2];
      if (curCity == dst) {
        if (citiesTillNow - 2 <= k) {
          minCost = Math.min(minCost, costTillNow);
        }
        continue;
      }
      for (int[] neighbours : graph.get(curCity)) {
        if (costTillNow + neighbours[1] < cost[neighbours[0]]
            && costTillNow + neighbours[1] < minCost
            && citiesTillNow + 1 - 2 <= k) {
          cost[neighbours[0]] = costTillNow + neighbours[1];
          queue.add(new int[]{neighbours[0], costTillNow + neighbours[1], citiesTillNow + 1});
        }
      }
    }
    return minCost == Integer.MAX_VALUE ? -1 : minCost;
  }

  public List<String> findItinerary(List<List<String>> tickets) {
    Map<String, PriorityQueue<String>> flights = new HashMap<>();
    for (List<String> ticket : tickets) {
      if (!flights.containsKey(ticket.get(0))) {
        flights.put(ticket.get(0), new PriorityQueue<>());
      }
      flights.get(ticket.get(0)).add(ticket.get(1));
    }

    LinkedList<String> curItinerary = new LinkedList<>();
    findItineraryUtil("JFK", flights, curItinerary);
    return curItinerary;
  }

  private void findItineraryUtil(String cur, Map<String, PriorityQueue<String>> map, LinkedList<String> curItinerary) {

    PriorityQueue<String> nextDepartures = map.get(cur);
    while (nextDepartures != null && !nextDepartures.isEmpty()) {
      findItineraryUtil(nextDepartures.poll(), map, curItinerary);
    }
    curItinerary.addFirst(cur);
  }

  public int[] findOrderUsingDFS(int n, int[][] prereq) {
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }
    for (int[] edge : prereq) {
      graph.get(edge[0]).add(edge[1]);
    }
    boolean[] visited = new boolean[n];
    Arrays.fill(visited, false);
    boolean[] currentPath = new boolean[n];
    Arrays.fill(currentPath, false);
    Deque<Integer> res = new LinkedList<>();
    for (int i = 0; i < n; i++) {
      if (visited[i]) {
        continue;
      }
      Stack<Integer> st = new Stack<>();
      st.add(i);
      visited[i] = true;
      currentPath[i] = true;
      while (!st.empty()) {
        Integer peek = st.peek();
        boolean unvisitedNeighbourFound = false;
        for (int neighbour : graph.get(peek)) {
          if (currentPath[neighbour]) {
            return new int[0];
          }
          if (!visited[neighbour]) {
            st.add(neighbour);
            visited[neighbour] = true;
            currentPath[neighbour] = true;
            unvisitedNeighbourFound = true;
            break;
          }
        }
        if (!unvisitedNeighbourFound) {
          currentPath[peek] = false;
          st.pop();
          res.addLast(peek);
        }
      }
    }
    return res.stream().mapToInt(value -> value).toArray();
  }

  public int[] findOrderUsingBFS(int n, int[][] prereq) {
    int[] inDegree = new int[n];
    Arrays.fill(inDegree, 0);
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }
    for (int[] edge : prereq) {
      graph.get(edge[1]).add(edge[0]);
      inDegree[edge[0]]++;
    }
    Queue<Integer> queue = new LinkedList<>();
    for (int i = 0; i < n; i++) {
      if (inDegree[i] == 0) {
        queue.add(i);
      }
    }
    int[] res = new int[n];
    int idx = 0;
    while (!queue.isEmpty()) {
      Integer polled = queue.poll();
      res[idx++] = polled;
      for (int neighbour : graph.get(polled)) {
        if (--inDegree[neighbour] == 0) {
          queue.add(neighbour);
        }
      }
    }
    if (idx == n) {
      return res;
    } else {
      return new int[0];
    }
  }

  // using dfs for topological sort
  public boolean canFinishUsingDFS(int n, int[][] prereq) {
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }
    for (int[] edge : prereq) {
      graph.get(edge[0]).add(edge[1]);
    }
    boolean[] visited = new boolean[n];
    Arrays.fill(visited, false);
    boolean[] currentPath = new boolean[n];
    Arrays.fill(currentPath, false);
    for (int i = 0; i < n; i++) {
      if (visited[i]) {
        continue;
      }
      Stack<Integer> st = new Stack<>();
      st.add(i);
      visited[i] = true;
      currentPath[i] = true;
      while (!st.empty()) {
        Integer peek = st.peek();
        boolean unvisitedNeighbourFound = false;
        for (int neighbour : graph.get(peek)) {
          if (currentPath[neighbour]) {
            return false;
          }
          if (!visited[neighbour]) {
            st.add(neighbour);
            visited[neighbour] = true;
            currentPath[neighbour] = true;
            unvisitedNeighbourFound = true;
            break;
          }
        }
        if (!unvisitedNeighbourFound) {
          currentPath[peek] = false;
          st.pop();
        }
      }
    }
    return true;
  }

  // using bfs to detect cycle. topological sort method.
  public boolean canFinishUsingBFS(int n, int[][] prereq) {
    int[] inDegree = new int[n];
    Arrays.fill(inDegree, 0);
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }
    for (int[] edge : prereq) {
      graph.get(edge[0]).add(edge[1]);
      inDegree[edge[1]]++;
    }
    Queue<Integer> queue = new LinkedList<>();
    for (int i = 0; i < n; i++) {
      if (inDegree[i] == 0) {
        queue.add(i);
      }
    }
    while (!queue.isEmpty()) {
      Integer polled = queue.poll();
      for (int neighbour : graph.get(polled)) {
        if (--inDegree[neighbour] == 0) {
          queue.add(neighbour);
        }
      }
    }
    for (int i = 0; i < n; i++) {
      if (inDegree[i] != 0) {
        return false;
      }
    }
    return true;
  }

  public List<Integer> findMinHeightTreesOptimized(int n, int[][] edges) {
    if (n == 1) {
      return Collections.singletonList(0);
    }
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }
    for (int[] edge : edges) {
      graph.get(edge[0]).add(edge[1]);
      graph.get(edge[1]).add(edge[0]);
    }
    List<Integer> leaves = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      if (graph.get(i).size() == 1) {
        leaves.add(i);
      }
    }
    while (n > 2) {
      n -= leaves.size();
      List<Integer> newLeaves = new ArrayList<>();
      for (Integer leaf : leaves) {
        for (int neighbour : graph.get(leaf)) {
          graph.get(neighbour).remove((Integer) leaf);
          if (graph.get(neighbour).size() == 1) {
            newLeaves.add(neighbour);
          }
        }
      }
      leaves.clear();
      leaves.addAll(newLeaves);
    }
    return leaves;
  }

  public List<Integer> findMinHeightTrees(int n, int[][] edges) {
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }
    for (int[] edge : edges) {
      graph.get(edge[0]).add(edge[1]);
      graph.get(edge[1]).add(edge[0]);
    }
    List<Integer> ans = new ArrayList<>();
    int minD = Integer.MAX_VALUE;

    Queue<Integer> bfsQueue = new LinkedList<>();
    boolean[] visited = new boolean[n];
    for (int i = 0; i < n; i++) {
      Arrays.fill(visited, false);
      bfsQueue.clear();
      bfsQueue.add(i);
      visited[i] = true;
      int curD = -1;
      while (!bfsQueue.isEmpty()) {
        curD++;
        List<Integer> tempList = new ArrayList<>();
        while (!bfsQueue.isEmpty()) {
          Integer polled = bfsQueue.poll();
          for (int neighbour : graph.get(polled)) {
            if (!visited[neighbour]) {
              visited[neighbour] = true;
              tempList.add(neighbour);
            }
          }
        }
        bfsQueue.addAll(tempList);
      }
      if (curD == minD) {
        ans.add(i);
      } else if (curD < minD) {
        minD = curD;
        ans.clear();
        ans.add(i);
      }
    }
    return ans;
  }

  public int removeStonesII(int[][] stones) {
    int[] root = new int[stones.length];
    for (int i = 0; i < root.length; i++) {
      root[i] = i;
    }
    for (int i = 0; i < stones.length; i++) {
      int x = stones[i][0], y = stones[i][1];
      for (int j = 0; j < stones.length; j++) {
        if (j == i) continue;
        if (stones[j][0] == x || stones[j][1] == y) {
          int p1 = findRoot(i, root);
          int p2 = findRoot(j, root);
          if (p1 != p2) {
            root[p1] = p2;
          }
        }
      }
    }
    Set<Integer> roots = new HashSet<>();
    for (int aRoot : root) {
      roots.add(findRoot(aRoot, root));
    }
    return stones.length - roots.size();
  }

  private int findRoot(int idx, int[] root) {
    while (root[idx] != idx) {
      root[idx] = root[root[idx]];
      idx = root[idx];
    }
    return idx;
  }

  private Set<MyPoint> visited;

  public int removeStones(int[][] stones) {
    List<MyPoint> points = new ArrayList<>();
    for (int[] stone : stones) {
      MyPoint point = new MyPoint(stone[0], stone[1]);
      for (MyPoint p : points) {
        if (p.x == point.x || p.y == point.y) {
          p.connectedPoints.add(point);
          point.connectedPoints.add(p);
        }
      }
      points.add(point);
    }
    visited = new HashSet<>();
    int islands = 0;
    for (MyPoint p : points) {
      if (!visited.contains(p)) {
        islands++;
        dfsStones(p);
      }
    }
    return points.size() - islands;
  }

  private void dfsStones(MyPoint curPoint) {
    visited.add(curPoint);
    for (int i = 0; i < curPoint.connectedPoints.size(); i++) {
      if (!visited.contains(curPoint.connectedPoints.get(i))) {
        dfsStones(curPoint.connectedPoints.get(i));
      }
    }
  }


  public Node cloneGraph(Node node) {
    if (node == null) {
      return node;
    }
    Map<Node, Node> map = new HashMap<>();
    Node cloned = new Node(node.val, new ArrayList<>());
    map.put(node, cloned);
    Queue<Node> st = new LinkedList<>();
    st.add(node);
    while (!st.isEmpty()) {
      Node t1 = st.poll(), t2 = map.get(t1);
      for (int i = 0; i < t1.neighbors.size(); i++) {
        Node curNeighbour = t1.neighbors.get(i);
        Node curClone;
        if (map.containsKey(curNeighbour)) {
          curClone = map.get(curNeighbour);
        } else {
          curClone = new Node(curNeighbour.val, new ArrayList<>());
          st.add(curNeighbour);
          map.put(curNeighbour, curClone);
        }
        t2.neighbors.add(curClone);
      }
    }
    return cloned;
  }


  class Node {
    public int val;
    public List<Node> neighbors;

    public Node() {
    }

    public Node(int _val, List<Node> _neighbors) {
      val = _val;
      neighbors = _neighbors;
    }
  }

  ;

  public int numIslands2(char[][] grid) {
    int m = grid.length;
    if (m == 0) {
      return 0;
    }
    int n = grid[0].length;
    int numIslands = 0;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (grid[i][j] == '1') {
          numIslands++;
          dfs(i, j, m, n, grid);
        }
      }
    }
    return numIslands;
  }

  public int numIslands(char[][] grid) {
    int m = grid.length;
    if (m == 0) {
      return 0;
    }
    int n = grid[0].length;
    int numIslands = 0;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (grid[i][j] == '1') {
          numIslands++;
          grid[i][j] = '0';
          Stack<Point> st = new Stack<>();
          st.push(new Point(i, j));
          while (!st.empty()) {
            Point pair = st.pop();
            int x = pair.x, y = pair.y;
            int[] ivalues = {-1, 0, 1, 0};
            int[] jvalues = {0, -1, 0, 1};
            for (int idx = 0; idx < 4; idx++) {
              int newX = x + ivalues[idx], newY = y + jvalues[idx];
              if (isValidPoint(m, n, newX, newY)
                  && grid[newX][newY] == '1') {
                grid[newX][newY] = '0';
                st.push(new Point(newX, newY));
              }
            }
          }
        }
      }
    }

    return numIslands;
  }

  private boolean isValidPoint(int m, int n, int newX, int newY) {
    return newX >= 0 && newY >= 0 && newX < m && newY < n;
  }

  private void dfs(int i, int j, int m, int n, char[][] grid) {
    if (!isValidPoint(m, n, i, j) || grid[i][j] == '0') {
      return;
    }
    grid[i][j] = '0';
    int[] ivalues = {-1, 0, 1, 0};
    int[] jvalues = {0, -1, 0, 1};
    for (int idx = 0; idx < 4; idx++) {
      int newX = i + ivalues[idx], newY = j + jvalues[idx];
      dfs(newX, newY, m, n, grid);
    }
  }

  public int countComponents(int n, int[][] edges) {
    ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }
    for (int[] edge : edges) {
      graph.get(edge[0]).add(edge[1]);
      graph.get(edge[1]).add(edge[0]);
    }
    int numComponents = 0;
    int[] visited = new int[n];
    intializeArray(n, visited, 0);
    for (int i = 0; i < n; i++) {
      if (visited[i] == 1) {
        continue;
      }
      numComponents++;
      Stack<Main.IntegerPair> st = new Stack<>();
      st.push(new Main.IntegerPair(0, -1));
      visited[0] = 1;
      while (!st.empty()) {
        Main.IntegerPair pair = st.pop();
        int t = pair.first;
        int parent = pair.second;
        ArrayList<Integer> neighbours = graph.get(t);
        for (int neighbour : neighbours) {
          if (neighbour != parent && visited[neighbour] == 0) {
            visited[neighbour] = 1;
            st.push(new Main.IntegerPair(neighbour, t));
          }
        }
      }
    }
    return numComponents;
  }

  public static boolean validTree2(int n, int[][] edges) {
    ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }
    for (int[] edge : edges) {
      graph.get(edge[0]).add(edge[1]);
      graph.get(edge[1]).add(edge[0]);
    }
    int[] visited = new int[n];
    intializeArray(n, visited, 0);
    boolean hasCycle = hasCycle(0, -1, visited, graph);
    if (hasCycle) {
      return false;
    }
    for (int i = 0; i < n; i++) {
      if (visited[i] == 0) {
        return false;
      }
    }
    return true;
  }

  private static boolean hasCycle(int u, int parent, int[] visited, ArrayList<ArrayList<Integer>> graph) {
    visited[u] = 1;
    ArrayList<Integer> curNeighbours = graph.get(u);
    for (int neighbour : curNeighbours) {
      if (visited[neighbour] == 1 && neighbour != parent) {
        return true;
      }
      if (visited[neighbour] == 0) {
        if (hasCycle(neighbour, u, visited, graph)) {
          return true;
        }
      }
    }
    return false;
  }

  public static boolean validTree(int n, int[][] edges) {
    ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }
    for (int[] edge : edges) {
      graph.get(edge[0]).add(edge[1]);
      graph.get(edge[1]).add(edge[0]);
    }
    int[] visited = new int[n];
    intializeArray(n, visited, 0);
    Stack<Main.IntegerPair> st = new Stack<>();
    st.push(new Main.IntegerPair(0, -1));
    visited[0] = 1;
    int[] currentPath = new int[n];
    intializeArray(n, currentPath, 0);
    currentPath[0] = 1;
    while (!st.isEmpty()) {
      Main.IntegerPair pair = st.peek();
      int t = pair.first;
      int parent = pair.second;
      System.out.println("peeked: " + t + " with parent: " + parent);
      boolean unvisitedNeighbourFound = false;
      ArrayList<Integer> curNeighbours = graph.get(t);
      for (int neighbour : curNeighbours) {
        if (neighbour != parent && visited[neighbour] == 1) {
          return false;
        }
        if (visited[neighbour] == 0) {
          unvisitedNeighbourFound = true;
          System.out.println("pushing: " + neighbour + " with parent: " + t);
          st.push(new Main.IntegerPair(neighbour, t));
          visited[neighbour] = 1;
          currentPath[neighbour] = 1;
          break;
        }
      }
      if (!unvisitedNeighbourFound) {
        System.out.println("popping");
        st.pop();
        currentPath[t] = 0;
      }
    }
    for (int i = 0; i < n; i++) {
      if (visited[i] == 0) {
        return false;
      }
    }
    return true;
  }

  public boolean canFinish(int n, int[][] prereq) {
    ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      graph.add(new ArrayList<>());
    }
    for (int[] aPrereq : prereq) {
      graph.get(aPrereq[0]).add(aPrereq[1]);
    }
    int[] visited = new int[n];
    intializeArray(n, visited, 0);
    for (int i = 0; i < n; i++) {
      if (visited[i] == 0) {
        Stack<Integer> st = new Stack<>();
        int[] tempVisited = new int[n];
        intializeArray(n, tempVisited, 0);
        st.push(i);
        visited[i] = 1;
        tempVisited[i] = 1;
        while (!st.isEmpty()) {
          int t = st.peek();
          boolean unvisitedNeighbourFound = false;
          ArrayList<Integer> curNeighbours = graph.get(t);
          for (int neighbour : curNeighbours) {
            if (tempVisited[neighbour] == 1) {
              return false;
            }
            if (visited[neighbour] == 0) {
              unvisitedNeighbourFound = true;
              st.push(neighbour);
              visited[neighbour] = 1;
              tempVisited[neighbour] = 1;
              break;
            }
          }
          if (!unvisitedNeighbourFound) {
            st.pop();
            tempVisited[t] = 0;
          }
        }
      }
    }
    return true;
  }

  private static void intializeArray(int n, int[] inDegree, int i2) {
    for (int i = 0; i < n; i++) {
      inDegree[i] = i2;
    }
  }

  class Point {
    int x, y;

    public Point(int x, int y) {
      this.x = x;
      this.y = y;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      Point point = (Point) o;
      return x == point.x &&
          y == point.y;
    }

    @Override
    public int hashCode() {
      return Objects.hash(x, y);
    }
  }
}
