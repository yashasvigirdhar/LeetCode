import java.util.*;

public class Graph {


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

  private int[][] dx = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

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

  class Pair<T> {
    T first, second;

    public Pair(T first, T second) {
      this.first = first;
      this.second = second;
    }
  }
}
