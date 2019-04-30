import java.util.*;

public class DP {


  /*
  Every station is a candidate of the final solution. The main thing is: When exactly does it becomes part of the solution ->
  When you can't proceed without it -> When you've tried all other stations with gas more than this candidate to proceed forward.
   */
  public int minRefuelStopsIIIGreedy(int target, int startFuel, int[][] stations) {
    int totalFuel = startFuel;
    int numRefuel = 0;
    int curStation = 0;
    PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.reverseOrder());

    while (totalFuel < target) {
      while (curStation < stations.length && totalFuel - stations[curStation][0] >= 0) {
        pq.add(stations[curStation][1]);
        curStation++;
      }
      if (pq.isEmpty()) {
        return -1;
      }
      totalFuel = totalFuel + pq.poll();
      numRefuel++;
    }
    return numRefuel;
  }

  // this is wrong. Wrong analogy to jump game II.  Unlike jump game where I maintain only one successor, here, I've
  // to maintain a list of successors. Since using only one successors might not be enough.
  // In jump game, When I had concluded that a particular idx is my successor, other candidates were useless. Here they are
  // not.
  // The basic difference is, all the candidate successors can add up to the result of best successor. In jump case,
  // candidate successors don't add up to each other's result.
  public int minRefuelStopsII(int target, int startFuel, int[][] stations) {

    int maxEndDistance = startFuel;
    int numRefuel = 0;
    int curStation = -1;
    while (maxEndDistance < target) {
      int nextStationToStart = curStation + 1;
      int nextEnd = -1;
      int idx = curStation + 1;
      while (idx < stations.length && maxEndDistance - stations[idx][0] >= 0) {
        int endDistanceWithThisRefuel = Math.max(stations[idx][0], maxEndDistance) + stations[idx][1];
        if (endDistanceWithThisRefuel > nextEnd) {
          nextEnd = endDistanceWithThisRefuel;
          nextStationToStart = idx;
        }
        idx++;
      }
      if (nextEnd == -1) {
        return -1;
      }
      curStation = nextStationToStart;
      maxEndDistance = nextEnd;
      numRefuel++;
    }
    return numRefuel;
  }

  HashMap<Integer, Integer>[] fuelDP;

  public int minRefuelStops(int target, int startFuel, int[][] stations) {
    if (stations.length == 0) {
      if (startFuel < target) {
        return -1;
      } else {
        return 0;
      }
    }
    if (startFuel < stations[0][0]) {
      return -1;
    }
    fuelDP = new HashMap[stations.length];
    return minRefuelStopsUtil(0, target, startFuel - stations[0][0], stations);
  }

  private int minRefuelStopsUtil(int idx, int target, int startFuel, int[][] stations) {
    if (idx == stations.length - 1) {
      if (stations[idx][0] + startFuel >= target) {
        return 0;
      } else if (stations[idx][0] + startFuel + stations[idx][1] >= target) {
        return 1;
      } else {
        return -1;
      }
    }
    if (fuelDP[idx] != null && fuelDP[idx].containsKey(startFuel)) {
      return fuelDP[idx].get(startFuel);
    }

    if (fuelDP[idx] == null) {
      fuelDP[idx] = new HashMap<>();
    }
    int ans = Integer.MAX_VALUE;

    // if I don't refuel
    for (int i = idx + 1; i < stations.length && startFuel + stations[idx][0] >= stations[i][0]; i++) {
      int nextStationDiff = stations[i][0] - stations[idx][0];
      int returned = minRefuelStopsUtil(i, target, startFuel - nextStationDiff, stations);
      if (returned != -1) {
        ans = Math.min(ans, returned);
      }
      if (ans == 0) {
        break;
      }
    }

    if (ans <= 1) {
      // can return here as refuelling here will also give at least 1 as ans
      fuelDP[idx].put(startFuel, ans);
      return fuelDP[idx].get(startFuel);
    }

    //refuelling
    for (int i = idx + 1; i < stations.length && stations[idx][0] + startFuel + stations[idx][1] >= stations[i][0]; i++) {
      int nextStationDiff = stations[i][0] - stations[idx][0];
      int returned = minRefuelStopsUtil(i, target, startFuel + stations[idx][1] - nextStationDiff, stations);
      if (returned != -1) {
        ans = Math.min(ans, returned + 1);
      }
      if (returned == 0) {
        break;
      }
    }


    if (ans == Integer.MAX_VALUE) {
      fuelDP[idx].put(startFuel, -1);
    } else {
      fuelDP[idx].put(startFuel, ans);
    }
    return fuelDP[idx].get(startFuel);
  }

  public int mincostTicketsPractice(int[] days, int[] costs) {
    int n = days.length;
    int[][] ticketsDP = new int[n][4];
    System.arraycopy(costs, 0, ticketsDP[0], 0, 3);
    ticketsDP[0][3] = Integer.MAX_VALUE;
    for (int i = 1; i < n; i++) {
      int prevMin = minInArray(ticketsDP[i - 1]);
      for (int j = 0; j < 3; j++) {
        ticketsDP[i][j] = prevMin + costs[j];
      }
      ticketsDP[i][3] = Integer.MAX_VALUE;
      int tillSeven = days[i] - 7 + 1;
      int tillThirty = days[i] - 30 + 1;
      int j = i - 1;
      while (j >= 0) {
        if (days[j] >= tillSeven) {
          ticketsDP[i][3] = Math.min(ticketsDP[i][3], ticketsDP[j][1]);
        }
        if (days[j] >= tillThirty) {
          ticketsDP[i][3] = Math.min(ticketsDP[i][3], ticketsDP[j][2]);
        } else {
          break;
        }
        j--;
      }
    }
    return minInArray(ticketsDP[n - 1]);
  }

  private int minInArray(int[] nums) {
    int min = nums[0];
    for (int n : nums) {
      min = Math.min(n, min);
    }
    return min;
  }

  public int jumpGameIIOrderN(int[] nums) {
    if (nums.length == 0 || nums.length == 1) {
      return 0;
    }
    int curStart = 0, curEnd = curStart + nums[0];
    int jump = 1;
    while (curEnd < nums.length - 1) {
      int nextStart = curEnd;
      for (int i = curStart + 1; i <= curEnd; i++) {
        if (i + nums[i] > curEnd) {
          nextStart = i;
        }
      }
      curStart = nextStart;
      curEnd = curStart + nums[curStart];
      jump++;
    }
    return jump;
  }

  public int findLengthMemoryOptimized(int[] A, int[] B) {
    int m = A.length;
    int n = B.length;
    int[] dp = new int[n];
    int max = 0;

    int[] temp = new int[n];
    for (int i = 0; i < m; i++) {
      Arrays.fill(temp, 0);
      for (int j = 0; j < n; j++) {
        if (A[i] == B[j]) {
          if (i == 0 || j == 0) {
            temp[j] = 1;
          } else {
            temp[j] = dp[j - 1] + 1;
          }
          max = Math.max(max, temp[j]);
        }
      }
      System.arraycopy(temp, 0, dp, 0, n);
    }
    return max;
  }

  public int findLength(int[] A, int[] B) {
    int m = A.length;
    int n = B.length;
    int[][] dp = new int[m][n];
    for (int[] arr : dp) Arrays.fill(arr, 0);
    int max = 0;
    for (int i = 0; i < m; i++) {
      if (A[i] == B[0]) {
        dp[i][0] = 1;
        max = 1;
      }
    }
    for (int i = 0; i < n; i++) {
      if (B[i] == A[0]) {
        dp[0][i] = 1;
        max = 1;
      }
    }
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        if (A[i] == B[j]) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
          max = Math.max(max, dp[i][j]);
        }
      }
    }
    return max;
  }


  public int twoCitySchedCostII(int[][] costs) {
    int n = costs.length / 2;
    int[][] dp = new int[n + 1][n + 1];
    dp[0][0] = 0;
    for (int i = 1; i <= n; i++) {
      dp[0][i] = dp[0][i - 1] + costs[i][1];
    }

    for (int i = 1; i <= n; i++) {
      dp[i][0] = dp[i - 1][0] + costs[i][0];
    }

    for (int i = 1; i <= n; i++) {
      for (int j = 1; j <= n; j++) {
        dp[i][j] = Math.min(dp[i - 1][j] + costs[i + j - 1][0], dp[i][j - 1] + costs[i + j - 1][1]);
      }
    }
    return dp[n][n];
  }

  private int[][][] dpCity;

  public int twoCitySchedCost(int[][] costs) {
    int n = costs.length;
    dpCity = new int[n][(n / 2) + 1][(n / 2) + 1];
    for (int[][] arr : dpCity) {
      for (int[] a : arr) {
        Arrays.fill(a, -1);
      }
    }
    return twoCitySchedCostUtil(0, costs, -1, 0, n / 2);
  }

  private int twoCitySchedCostUtil(int idx, int[][] costs, int aCount, int bCount, int n) {
    if (idx == costs.length) {
      return 0;
    }
    if (dpCity[idx][aCount][bCount] != -1) {
      return dpCity[idx][aCount][bCount];
    }
    int a = Integer.MAX_VALUE, b = Integer.MAX_VALUE;
    if (aCount < n) {
      a = twoCitySchedCostUtil(idx + 1, costs, aCount + 1, bCount, n) + costs[idx][0];
    }
    if (bCount < n) {
      b = twoCitySchedCostUtil(idx + 1, costs, aCount, bCount + 1, n) + costs[idx][1];
    }
    dpCity[idx][aCount][bCount] = Math.min(a, b);
    return dpCity[idx][aCount][bCount];
  }

  public int longestLine(int[][] b) {
    int m = b.length;
    if (m == 0) return 0;
    int n = b[0].length;
    int[][][] dp = new int[m][n][4];
    int max;

    max = dp[0][0][0] = dp[0][0][1] = dp[0][0][2] = dp[0][0][3] = b[0][0];

    for (int i = 1; i < m; i++) {
      if (b[i][0] == 1) {
        dp[i][0][1] = dp[i - 1][0][1] + 1;
        dp[i][0][3] = dp[i][0][2] = dp[i][0][0] = 1;
        max = Math.max(max, dp[i][0][1]);
      } else {
        dp[i][0][0] = dp[i][0][1] = dp[i][0][2] = dp[i][0][3] = 0;
      }
    }

    for (int i = 1; i < n; i++) {
      if (b[0][i] == 1) {
        dp[0][i][0] = dp[0][i - 1][0] + 1;
        dp[0][i][3] = dp[0][i][2] = dp[0][i][1] = 1;
        max = Math.max(max, dp[0][i][0]);
      } else {
        dp[0][i][0] = dp[0][i][1] = dp[0][i][2] = dp[0][i][3] = 0;
      }
    }

    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        if (b[i][j] == 1) {
          dp[i][j][0] = dp[i][j - 1][0] + 1;
          dp[i][j][1] = dp[i - 1][j][1] + 1;
          dp[i][j][2] = dp[i - 1][j - 1][2] + 1;
          if (j + 1 < n) {
            dp[i][j][3] = dp[i - 1][j + 1][3] + 1;
          } else {
            dp[i][j][3] = 1;
          }
          for (int k : dp[i][j]) {
            max = Math.max(max, k);
          }
        } else {
          dp[i][j][0] = dp[i][j][1] = dp[i][j][2] = dp[i][j][3] = 0;
        }
      }
    }
    for (int i = 1; i < m; i++) {
      if (b[i][0] == 1) {
        if (1 < n) {
          dp[i][0][3] = dp[i - 1][1][3] + 1;
          max = Math.max(max, dp[i][0][3]);
        } else {
          dp[i][0][3] = 1;
          max = Math.max(max, 1);
        }
      }
    }
    return max;
  }

  public int jumpWithForwardDP(int[] nums) {
    int n = nums.length;
    if (n == 0) {
      return 0;
    }
    int[] dp = new int[n];
    Arrays.fill(dp, Integer.MAX_VALUE);
    dp[0] = 0;
    for (int i = 0; i < n; i++) {
      if (dp[i] == Integer.MAX_VALUE) {
        continue;
      }
      for (int j = i + 1; j <= i + nums[i] && j < n; j++) {
        dp[j] = Math.min(1 + dp[i], dp[j]);
        if (j == n - 1) {
          return dp[j];
        }
      }
    }
    return dp[0];
  }

  public int jump(int[] nums) {
    int n = nums.length;
    if (n == 0) {
      return 0;
    }
    int[] dp = new int[n];
    dp[n - 1] = 0;
    for (int i = n - 2; i >= 0; i--) {
      dp[i] = Integer.MAX_VALUE;
      for (int j = i + 1; j <= i + nums[i] && j < n; j++) {
        if (dp[j] >= 0) {
          dp[i] = Math.min(dp[i], 1 + dp[j]);
        }
      }
    }
    return dp[0];
  }

  public boolean canJump(int[] nums) {
    int n = nums.length;
    if (n == 0) {
      return false;
    }
    int[] dp = new int[n];
    dp[n - 1] = 0;
    for (int i = n - 2; i >= 0; i--) {
      dp[i] = 0;
      for (int j = i + 1; j <= i + nums[i] && j < n; j++) {
        if (dp[j] == 1) {
          dp[i] = 1;
          break;
        }
      }
    }
    return dp[0] == 1;
  }

  int[][][] stocksDP;

  public int maxProfitWithKTransactions(int k, int[] prices) {
    if (k > prices.length / 2) {
      return maxProfitWithMultipleTransactions(prices);
    }
    stocksDP = new int[prices.length][k + 1][2];
    for (int i = 0; i < prices.length; i++) {
      for (int j = 0; j < k + 1; j++) {
        Arrays.fill(stocksDP[i][j], -1);
      }
    }
    return maxProfitWith2TransactionsUtil(0, 0, 2, 0, prices);
  }

  public int maxProfitWith2TransactionsUtil(int idx, int curT, int capacityT, int curBought, int[] prices) {
    if (idx == prices.length) {
      return 0;
    }
    if (stocksDP[idx][curT][curBought] != -1) {
      return stocksDP[idx][curT][curBought];
    }

    int ans = 0;
    if (curBought == 1) {
      ans = Math.max(ans, maxProfitWith2TransactionsUtil(idx + 1, curT, capacityT, 0, prices) + prices[idx]); // sell it
      ans = Math.max(ans, maxProfitWith2TransactionsUtil(idx + 1, curT, capacityT, curBought, prices));  //rest
      // either sell it or rest.
    } else {
      if (curT == capacityT) {
        return 0;
      }
      ans = Math.max(ans, maxProfitWith2TransactionsUtil(idx + 1, curT + 1, capacityT, 1, prices) - prices[idx]);
      ans = Math.max(ans, maxProfitWith2TransactionsUtil(idx + 1, curT, capacityT, curBought, prices)); //rest
      // either buy new or rest.
    }
    stocksDP[idx][curT][curBought] = ans;
    return ans;
  }


  public int maxProfitWithCooldown_SpaceOptimized(int[] prices) {
    if (prices.length <= 1) {
      return 0;
    }
    int[][] dp = new int[2][2];
    dp[0][0] = 0;
    dp[0][1] = -prices[0];
    dp[1][0] = Math.max(dp[0][0], dp[0][1] + prices[1]);
    dp[1][1] = Math.max(dp[0][1], -prices[1]);
    for (int i = 2; i < prices.length; i++) {
      int[] temp = new int[2];
      temp[0] = Math.max(dp[1][0], dp[1][1] + prices[i]);
      temp[1] = Math.max(dp[1][1], dp[0][0] - prices[i]);
      System.arraycopy(dp[1], 0, dp[0], 0, 2);
      System.arraycopy(temp, 0, dp[1], 0, 2);
    }
    return Math.max(dp[1][0], dp[1][1]);
  }

  public int maxProfitWithCooldown(int[] prices) {
    if (prices.length <= 1) {
      return 0;
    }
    int[][] dp = new int[prices.length][2];
    dp[0][0] = 0;
    dp[0][1] = -prices[0];
    dp[1][0] = Math.max(dp[0][0], dp[0][1] + prices[1]);
    dp[1][1] = Math.max(dp[0][1], -prices[1]);
    for (int i = 2; i < prices.length; i++) {
      dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
      dp[i][1] = Math.max(dp[i - 1][1], dp[i - 2][0] - prices[i]);
    }
    return Math.max(dp[prices.length - 1][0], dp[prices.length - 1][1]);
  }

  public int maxProfitWithFees(int[] prices, int fee) {
    if (prices.length == 0) {
      return 0;
    }
    int[][] dp = new int[prices.length][2];
    dp[0][0] = 0;
    dp[0][1] = -prices[0] - fee;
    for (int i = 1; i < prices.length; i++) {
      dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
      dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i] - fee);
    }
    return Math.max(dp[prices.length - 1][0], dp[prices.length - 1][1]);
  }


  public int maxProfitWithMultipleTransactions(int[] prices) {
    if (prices.length == 0) {
      return 0;
    }
    int[][] dp = new int[prices.length][2];
    dp[0][0] = 0;
    dp[0][1] = -prices[0];
    for (int i = 1; i < prices.length; i++) {
      dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
      dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
    }
    return Math.max(dp[prices.length][0], dp[prices.length][1]);
  }

  int[] xx = new int[]{-1, 0, 1, 0};
  int[] yy = new int[]{0, 1, 0, -1};

  int[][] dp10;

  public int longestIncreasingPath(int[][] matrix) {
    int m = matrix.length;
    if (m == 0) return 0;
    int n = matrix[0].length;
    dp10 = new int[m][n];
    for (int[] a : dp10) Arrays.fill(a, -1);
    int ans = 1;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        boolean[][] visited = new boolean[m][n];
        for (boolean[] a : visited) Arrays.fill(a, false);
        ans = Math.max(ans, longestIncreasingPathUtil(i, j, matrix));
      }
    }
    return ans;
  }

  private int longestIncreasingPathUtil(int idx, int jdx, int[][] matrix) {
    if (dp10[idx][jdx] != -1) {
      return dp10[idx][jdx];
    }
    int ans = 0;
    for (int i = 0; i < 4; i++) {
      int newIdx = idx + xx[i];
      int newJdx = jdx + yy[i];
      if (isValid(newIdx, newJdx, matrix.length, matrix[0].length) && matrix[newIdx][newJdx] > matrix[idx][jdx]) {
        ans = Math.max(ans, longestIncreasingPathUtil(newIdx, newJdx, matrix));
      }
    }
    dp10[idx][jdx] = ans + 1;
    return dp10[idx][jdx];
  }

  private boolean isValid(int i, int j, int m, int n) {
    return i >= 0 && j >= 0 && i < m && j < n;
  }

  public int longestArithSeqLengthDP(int[] A) {
    HashMap<Integer, Integer>[] map = new HashMap[A.length];
    int ans = 0;
    for (int i = 1; i < A.length; i++) {
      for (int j = 0; j < i; j++) {
        int diff = A[i] - A[j];
        int curAns = 2;
        if (map[j] != null && map[j].containsKey(diff)) {
          curAns = Math.max(curAns, 1 + map[j].get(diff));
        }
        if (map[i] == null) {
          map[i] = new HashMap<>();
        }
        map[i].put(diff, curAns);
        ans = Math.max(ans, curAns);
      }
    }
    return ans;
  }


  public int longestArithSeqLengthBruteForce(int[] A) {
    int ans = 2;
    for (int i = 0; i < A.length; i++) {
      for (int j = i + 1; j < A.length; j++) {
        int diff = A[j] - A[i];
        int t = A[j];
        int count = 2;
        for (int k = j + 1; k < A.length; k++) {
          if (A[k] - t == diff) {
            count++;
            t = A[k];
          }
        }
        ans = Math.max(ans, count);
      }
    }
    return ans;
  }

  public int coinChange(int[] coins, int amount) {

    ArrayList<Integer> dp = new ArrayList<>();
    for (long i = 0; i <= amount; i++) {
      dp.add(Integer.MAX_VALUE);
    }
    dp.set(0, 0);
    for (int i = 1; i <= amount; i++) {
      for (int coin : coins) {
        if (i - coin >= 0) {
          dp.set(i, Math.min(dp.get(i), dp.get(i - coin) + 1));
        }
      }
    }
    return dp.get(amount) == Integer.MAX_VALUE ? -1 : dp.get(amount);
  }

  public int numWaysIII(int n, int k) {
    if (n == 0 || k == 0) return 0;
    if (n == 1) return k;
    int[] dp = new int[n + 1];
    dp[1] = k;
    dp[2] = k * k;
    for (int i = 3; i <= n; i++) {
      dp[i] = dp[i - 1] * k - dp[i - 2];
    }
    return dp[n];
  }

  public int countPalindromicSubsequences(String s) {
    int mod = 1000000007;
    Set<String> set = new HashSet<>();
    for (int i = 0; i < s.length(); i++) {
      set.add(s.substring(i, i + 1));
    }
    for (int i = 0; i < s.length() - 1; i++) {
      if (s.charAt(i) == s.charAt(i + 1)) {
        set.add(s.substring(i, i + 2));
      }
    }
    for (int size = 3; size <= s.length(); size++) {
      for (int start = 0; start <= s.length() - size; start++) {
        int end = start + size - 1;
        for (int i = start + 1; i < end; i++) {
          if (s.charAt(start) == s.charAt(i)) {
            set.add(s.substring(start, start + 1) + s.substring(i, i + 1));
          }
        }
        for (int i = start + 1; i < end; i++) {
          if (s.charAt(i) == s.charAt(end)) {
            set.add(s.substring(i, i + 1) + s.substring(end, end + 1));
          }
        }
        if (s.charAt(start) == s.charAt(end)) {
          set.add(s.substring(start, end + 1));
          set.add(s.substring(start, start + 1) + s.substring(end, end + 1));
        }
      }
    }
    return set.size() % mod;
  }

  public int longestPalindromeSubseqII(String s) {
    int[][] dp = new int[s.length()][s.length()];
    for (int[] a : dp) Arrays.fill(a, 0);
    for (int i = 0; i < s.length(); i++) {
      dp[i][i] = 1;
    }
    for (int i = 1; i < s.length(); i++) {
      for (int j = i - 1; j >= 0; j--) {
        if (s.charAt(i) == s.charAt(j)) {
          dp[i][j] = dp[i + 1][j - 1] + 2;
        } else {
          dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
        }
      }
    }
    return dp[s.length() - 1][0];
  }

  public int longestPalindromeSubseq(String s) {
    int[][] dp = new int[s.length()][s.length()];
    for (int i = 0; i < s.length(); i++) {
      dp[i][i] = 1;
    }
    for (int i = 0; i < s.length() - 1; i++) {
      if (s.charAt(i) == s.charAt(i + 1)) {
        dp[i][i + 1] = 2;
      } else {
        dp[i][i + 1] = 1;
      }
    }
    for (int size = 3; size <= s.length(); size++) {
      for (int start = 0; start <= s.length() - size; start++) {
        int end = start + size - 1;
        if (s.charAt(start) == s.charAt(end)) {
          dp[start][end] = dp[start + 1][end - 1] + 2;
        } else {
          dp[start][end] = Math.max(dp[start + 1][end], dp[start][end - 1]);
        }
      }
    }
    return dp[0][s.length() - 1];
  }

  public int countSubstrings(String s) {
    int ans = 0;
    char[] c = s.toCharArray();
    for (int i = 0; i < c.length; i++) {
      ans++;
      int l = i - 1, r = i + 1;
      while (l >= 0 && r < c.length && c[l] == c[r]) {
        ans++;
        l--;
        r++;
      }
    }
    for (int i = 0; i < c.length - 1; i++) {
      if (c[i] == c[i + 1]) {
        ans++;
        int l = i - 1, r = i + 2;
        while (l >= 0 && r < c.length && c[l] == c[r]) {
          ans++;
          l--;
          r++;
        }
      }
    }
    return ans;
  }

  public int numWaysII(int n, int k) {
    if (n == 0 || k == 0) return 0;
    if (n == 1) return k;
    int[] dp = new int[n + 1];
    dp[1] = k;
    dp[2] = k * k;
    for (int i = 3; i <= n; i++) {
      dp[i] = dp[i - 1] * (k - 1) + dp[i - 2] * (k - 1);
    }
    return dp[n];
  }

  public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
    return shoppingOffersUtil(price, special, needs, 0);
  }


  public int shoppingOffersUtil(List<Integer> price, List<List<Integer>> special, List<Integer> needs,
                                int pos) {
    // if offer is valid, choose to consider it
    // it not, proceed ahead.
    int minima = 0;
    for (int i = 0; i < needs.size(); i++) {
      minima += needs.get(i) * price.get(i);
    }
    for (int idx = pos; idx < special.size(); idx++) {
      List<Integer> curOffer = special.get(idx);
      boolean validOffer = true;
      for (int i = 0; i < needs.size(); i++) {
        if (curOffer.get(i) > needs.get(i)) {
          // offer not valid
          validOffer = false;
          break;
        }
      }
      if (!validOffer) {
        continue;
      }
      List<Integer> newNeeds = new ArrayList<>();
      for (int i = 0; i < needs.size(); i++) {
        newNeeds.add(needs.get(i) - curOffer.get(i));
      }
      int returnedPrice = shoppingOffersUtil(price, special, newNeeds, idx);
      int offerPrice = curOffer.get(curOffer.size() - 1);
      int priceUsed = Math.min(minima, offerPrice);
      minima = Math.min(minima, priceUsed + returnedPrice);
    }


    return minima;
  }


  public int minFallingPathSum(int[][] A) {
    int m = A.length;
    if (m == 0) {
      return 0;
    }

    int n = A[0].length;
    int min[] = new int[n];
    System.arraycopy(A[0], 0, min, 0, n);

    for (int i = 1; i < m; i++) {
      int[] curMin = new int[n];
      for (int j = 0; j < n; j++) {
        int leftIdx = Math.max(0, j - 1);
        int rightIdx = Math.min(n - 1, j + 1);
        curMin[j] = A[i][j] + Math.min(min[leftIdx], Math.min(min[j], min[rightIdx]));
      }
      System.arraycopy(curMin, 0, min, 0, n);
    }
    int ans = min[0];
    for (int i = 1; i < n; i++) {
      ans = Math.min(ans, min[i]);
    }
    return ans;
  }

  public int mincostTicketsII(int[] days, int[] costs) {
    boolean[] isTravelDay = new boolean[days.length];
    Arrays.fill(isTravelDay, false);
    for (int day : days) {
      isTravelDay[day] = true;
    }
    int[] minCost = new int[366];
    minCost[0] = 0;
    for (int i = 1; i < 366; i++) {
      if (!isTravelDay[i]) {
        minCost[i] = minCost[i - 1];
      }
      minCost[i] = Integer.MAX_VALUE;
      minCost[i] = Math.min(minCost[i], minCost[i - 1] + costs[0]);
      minCost[i] = Math.min(minCost[i], minCost[Math.max(0, i - 7)] + costs[1]);
      minCost[i] = Math.min(minCost[i], minCost[Math.max(0, i - 30)] + costs[2]);
    }
    return minCost[365];
  }

  private int[][][] ticketdp;

  public int mincostTickets(int[] days, int[] costs) {
    ticketdp = new int[days.length][days.length][costs.length];
    for (int[][] arr : ticketdp) {
      for (int[] a : arr) {
        Arrays.fill(a, -1);
      }
    }
    return mincostTicketsUtil(0, -1, -1, days, costs);
  }

  public int mincostTicketsUtil(int idx, int passBoughtIdx, int passType, int[] days, int[] costs) {

    if (idx == days.length - 1) {
      if (isPassValid(idx, passBoughtIdx, days, passType)) {
        return 0;
      } else {
        return Math.min(costs[0], Math.min(costs[1], costs[2]));
      }
    }
    if (passBoughtIdx != -1 && ticketdp[idx][passBoughtIdx][passType] != -1) {
      return ticketdp[idx][passBoughtIdx][passType];
    }

    int a = costs[0] + mincostTicketsUtil(idx + 1, idx, 0, days, costs);
    int b = costs[1] + mincostTicketsUtil(idx + 1, idx, 1, days, costs);
    int c = costs[2] + mincostTicketsUtil(idx + 1, idx, 2, days, costs);
    int ans = Math.min(a, Math.min(b, c));
    if (isPassValid(idx, passBoughtIdx, days, passType)) {
      int d = mincostTicketsUtil(idx + 1, passBoughtIdx, passType, days, costs);
      ans = Math.min(ans, d);
    }
    if (passBoughtIdx != -1) {
      ticketdp[idx][passBoughtIdx][passType] = ans;
    }
    return ans;
  }

  private boolean isPassValid(int curIdx, int passBoughtIdx, int[] days, int passType) {
    if (passBoughtIdx == -1) {
      return false;
    }
    int daysDiff = days[curIdx] - days[passBoughtIdx];
    return (passType == 0 && daysDiff < 1)
            || (passType == 1 && daysDiff < 7)
            || (passType == 2 && daysDiff < 30);
  }

  public int rob(int[] nums) {
    if (nums.length == 0) return 0;
    int m1 = 0, m2 = nums[0];
    for (int i = 1; i < nums.length; i++) {
      int newM1 = Math.max(m1, m2);
      m2 = m1 + nums[i];
      m1 = newM1;
    }
    return Math.max(m1, m2);
  }

  private Map<Integer, List<String>> dp3 = new HashMap<>();
  int maxLen = 0;

  public List<String> wordBreak2(String s, List<String> wordDict) {
    for (String str : wordDict) {
      maxLen = Math.max(str.length(), maxLen);
    }
    return wordBreak2(0, s, wordDict);
  }


  private List<String> wordBreak2(int idx, String s, List<String> wordDict) {

    if (dp3.containsKey(idx)) {
      return dp3.get(idx);
    }

    List<String> curResult = new ArrayList<>();

    for (int i = idx; i < s.length() && i < idx + maxLen; i++) {
      String curStr = s.substring(idx, i + 1);
      boolean existsInDict = doesExistInDict(wordDict, curStr);
      if (existsInDict) {
        if (i == s.length()) {
          curResult.add(curStr);
        } else {
          List<String> res = wordBreak2(i + 1, s, wordDict);
          for (String re : res) {
            curResult.add(curStr + " " + re);
          }

        }
      }
    }
    dp3.put(idx, curResult);
    return curResult;
  }


  public boolean wordBreakII(String s, List<String> wordDict) {
    int n = s.length();
    boolean[] isPresent = new boolean[n];
    for (int i = 0; i < n; i++) isPresent[i] = false;

    for (int i = 0; i < n; i++) {
      if (doesExistInDict(wordDict, s.substring(0, i + 1))) {
        isPresent[i] = true;
        continue;
      }
      for (int j = i - 1; j >= 0; j--) {
        if (isPresent[j] && doesExistInDict(wordDict, s.substring(j + 1, i + 1))) {
          isPresent[i] = true;
        }
      }
    }
    return isPresent[n - 1];
  }

  private int[] dp1D;

  public boolean wordBreak(String s, List<String> wordDict) {
    dp1D = new int[s.length()];
    for (int i = 0; i < s.length(); i++) {
      dp1D[i] = -1;
    }
    return wordBreakUtil(0, s, wordDict);
  }

  public boolean wordBreakUtil(int idx, String s, List<String> wordDict) {
    if (idx == s.length()) {
      return true;
    }
    if (dp1D[idx] != -1) {
      return (dp1D[idx] == 1);
    }
    for (int i = idx; i < s.length(); i++) {
      boolean existsInDict;
      existsInDict = doesExistInDict(wordDict, s.substring(idx, i + 1));
      if (existsInDict) {
        if (wordBreakUtil(idx + 1, s, wordDict)) {
          dp1D[idx] = 1;
          return true;
        }
      }
    }
    dp1D[idx] = 0;
    return false;
  }


  private boolean doesExistInDict(List<String> wordDict, String curStr) {
    for (String word : wordDict) {
      if (word.equals(curStr)) {
        return true;
      }
    }
    return false;
  }

  public int eraseOverlapIntervals2(Interval[] intervals) {
    int n = intervals.length;
    Arrays.sort(intervals, Comparator.comparingInt(o -> o.start));
    int[] dp = new int[n];
    for (int i = 0; i < n; i++) {
      dp[i] = 1;
    }
    int max = 1;
    for (int i = 1; i < n; i++) {
      for (int j = 0; j < i; j++) {
        if (intervals[i].start >= intervals[j].end && dp[i] < dp[j] + 1) {
          dp[i] = dp[j] + 1;
          max = Math.max(max, dp[i]);
        }
      }
    }
    return n - max;
  }

  private int[][] dp;

  public int eraseOverlapIntervals(Interval[] intervals) {
    int n = intervals.length;
    if (n == 0) return 0;
    dp = new int[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        dp[i][j] = -1;
      }
    }
    Arrays.sort(intervals, Comparator.comparingInt(o -> o.start));
    return eraseOverlapIntervalUtil(-1, 0, intervals);
  }


  private int eraseOverlapIntervalUtil(int prev, int cur, Interval[] intervals) {
    if (cur == intervals.length - 1) {
      if (prev == -1 || intervals[prev].end <= intervals[cur].start) {
        return 0;
      } else {
        return 1;
      }
    }
    if (prev != -1 && dp[prev][cur] != -1) {
      return dp[prev][cur];
    }
    int taken = Integer.MAX_VALUE;
    if (prev == -1 || intervals[prev].end <= intervals[cur].start) {
      taken = eraseOverlapIntervalUtil(cur, cur + 1, intervals);
    }
    int notTaken = eraseOverlapIntervalUtil(prev, cur + 1, intervals) + 1;
    if (prev == -1) return Math.min(taken, notTaken);
    dp[prev][cur] = Math.min(taken, notTaken);
    return dp[prev][cur];
  }


  public int wiggleMaxLength(int[] nums) {
    int n = nums.length;
    int[][] dp = new int[n][2];
    for (int i = 0; i < n; i++) {
      dp[i][0] = 1;
      dp[i][1] = 1;
    }
    for (int i = 1; i < n; i++) {
      for (int j = 0; j < i; j++) {
        if (nums[i] > nums[j] && dp[i][0] < dp[j][1] + 1) {
          dp[i][0] = dp[j][1] + 1;
        }
        if (nums[i] < nums[j] && dp[i][1] < dp[j][0] + 1) {
          dp[i][1] = dp[j][0] + 1;
        }
      }
    }
    return Math.max(dp[n - 1][0], dp[n - 1][1]);
  }

  private HashMap<String, Boolean> states;

  public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
    if (maxChoosableInteger >= desiredTotal) {
      return true;
    }
    if ((maxChoosableInteger + 1) * maxChoosableInteger / 2 < desiredTotal) {
      return false;
    }
    states = new HashMap<>();
    boolean[] chosen = new boolean[maxChoosableInteger + 1];
    for (int i = 0; i < chosen.length; i++) {
      chosen[i] = false;
    }
    return canIWinUtil(desiredTotal, chosen);
  }


  private boolean canIWinUtil(int sumLeft, boolean[] chosen) {
    if (sumLeft <= 0) {
      return false;
    }
    String stateKey = Arrays.toString(chosen);
    if (states.containsKey(stateKey)) {
      return states.get(stateKey);
    }
    for (int i = 1; i < chosen.length; i++) {
      if (!chosen[i]) {
        chosen[i] = true;
        if (!canIWinUtil(sumLeft - i, chosen)) {
          states.put(stateKey, true);
          chosen[i] = false;
          return true;
        }
        chosen[i] = false;
      }
    }
    states.put(stateKey, false);
    return false;
  }

  private int[][][] dp2;

  public int numWays(int n, int k) {
    dp2 = new int[n][k][3];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j++) {
        dp2[i][j][0] = -1;
        dp2[i][j][1] = -1;
        dp2[i][j][2] = -1;
      }
    }
    return numWaysUtil(0, -1, 0, n, k);
  }

  private int numWaysUtil(int curIdx, int lastColor, int lastColorCount, int n, int c) {
    if (curIdx == n) {
      return 1;
    }
    if (lastColor != -1 && dp2[curIdx][lastColor][lastColorCount] != -1) {
      return dp2[curIdx][lastColor][lastColorCount];
    }
    int total = 0;
    for (int idx = 0; idx < c; idx++) {
      if (idx == lastColor) {
        if (lastColorCount == 1) {
          total += numWaysUtil(curIdx + 1, idx, lastColorCount + 1, n, c);
        }
      } else {
        total += numWaysUtil(curIdx + 1, idx, 1, n, c);
      }
    }
    if (lastColor == -1) {
      return total;
    }
    dp2[curIdx][lastColor][lastColorCount] = total;
    return dp2[curIdx][lastColor][lastColorCount];
  }

  public int minCostII(int[][] costs) {
    int min1 = 0, min2 = 0, min1Idx = -1;
    for (int i = 0; i < costs.length; i++) {
      int curMin1 = Integer.MAX_VALUE, curMin2 = Integer.MAX_VALUE, curMinIndex = -1;
      for (int j = 0; j < costs[i].length; j++) {
        int cost = costs[i][j] + ((j == min1Idx) ? min2 : min1);
        if (cost < curMin1) {
          curMin2 = curMin1;
          curMin1 = cost;
          curMinIndex = j;
        } else if (cost < curMin2) {
          curMin2 = cost;
        }
      }
      min1 = curMin1;
      min2 = curMin2;
      min1Idx = curMinIndex;
    }
    return min1;
  }

  public int minCost2(int[][] costs) {
    int n = costs.length;
    if (n == 0) return 0;
    int[] prev = new int[3];
    prev[0] = costs[0][0];
    prev[1] = costs[0][1];
    prev[2] = costs[0][2];
    int[] cur = new int[3];
    for (int i = 1; i < n; i++) {
      cur[0] = Math.min(prev[1], prev[2]) + costs[i][0];
      cur[1] = Math.min(prev[0], prev[2]) + costs[i][1];
      cur[2] = Math.min(prev[0], prev[1]) + costs[i][2];
      System.arraycopy(cur, 0, prev, 0, 3);
    }
    return Math.min(prev[0], Math.min(prev[1], prev[2]));
  }


//  private int[][] dp;

  public int minCost(int[][] costs) {
    int n = costs.length;
    dp = new int[n][dp[0].length];
    initializeArray(dp, -1);
    return calc(0, costs, -1);
  }

  private void initializeArray(int[][] dp, int val) {
    for (int i = 0; i < dp.length; i++) {
      for (int j = 0; j < dp[0].length; j++) {
        dp[i][j] = val;
      }
    }
  }

  private int calc(int idx, int[][] costs, int prev) {
    if (prev != -1 && dp[idx][prev] != -1) {
      return dp[idx][prev];
    }
    if (idx == costs.length) {
      return 0;
    }
    int curCost = Integer.MAX_VALUE;
    for (int i = 0; i < costs[idx].length; i++) {
      if (i != prev) {
        curCost = Math.min(curCost, calc(idx + 1, costs, i) + costs[idx][i]);
      }
    }
    dp[idx][prev] = curCost;
    return dp[idx][prev];
  }
}
