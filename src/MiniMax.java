import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class MiniMax {

  int[][][] dp2;

  public boolean PredictTheWinnerII(int[] nums) {
    int n = nums.length;
    int[][] dp = new int[n][n];
    for (int i = 0; i < n; i++) {
      dp[i][i] = nums[i];
    }
    for (int size = 2; size <= n; size++) {
      for (int i = 0; i <= n - size; i++) {
        int j = i + size - 1;
        dp[i][j] = Math.max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1]);
      }
    }
    return dp[0][n - 1] >= 0;
  }

  public boolean PredictTheWinner(int[] nums) {
    int n = nums.length;
    dp2 = new int[n][n][2];
    for (int[][] arr : dp2)
      for (int[] subArr : arr)
        Arrays.fill(subArr, -1);
    return PredictTheWinnerUtil(0, n - 1, nums, 1) >= 0;
  }

  private int PredictTheWinnerUtil(int start, int end, int[] nums, int id) {
    if (start > end) {
      return 0;
    }
    if (dp[start][end][id] != -1) {
      return dp[start][end][id];
    }
    if (id == 1) {
      dp[start][end][id] = Math.max(nums[start] + PredictTheWinnerUtil(start + 1, end, nums, 0),
              nums[end] + PredictTheWinnerUtil(start, end - 1, nums, 0));
    } else {
      dp[start][end][id] = Math.max(-nums[start] + PredictTheWinnerUtil(start + 1, end, nums, 1),
              -nums[end] + PredictTheWinnerUtil(start, end - 1, nums, 1));
    }
    return dp[start][end][id];
  }

  private Map<String, Boolean> stateMap;

  public boolean canWin(String s) {
    stateMap = new HashMap<>();
    return canWin(s.toCharArray());
  }

  public boolean canWin(char[] s) {
    String key = Arrays.toString(s);
    if (stateMap.containsKey(key)) {
      return stateMap.get(key);
    }
    for (int i = 0; i < s.length - 1; i++) {
      if (s[i] == '+' && s[i + 1] == '+') {
        s[i] = '-';
        s[i + 1] = '-';
        if (!canWin(s)) {
          s[i] = '+';
          s[i + 1] = '+';
          stateMap.put(key, true);
          return true;
        }
        s[i] = '+';
        s[i + 1] = '+';
      }
    }
    stateMap.put(key, false);
    return false;
  }

  public boolean canWinNimII(int n) {
    return n % 4 != 0;
  }

  public boolean canWinNim(int n) {
    boolean p1 = true, p2 = true, p3 = true;
    for (int i = 4; i <= n; i++) {
      if (!p1 || !p2 || !p3) {
        p1 = p2;
        p2 = p3;
        p3 = true;
      } else {
        p1 = p2;
        p2 = p3;
        p3 = false;
      }
    }
    return p3;
  }

  public boolean stoneGameII(int[] nums) {
    int n = nums.length;
    int[][] dp = new int[n][n];
    for (int i = 0; i < n; i++) {
      dp[i][i] = nums[i];
    }
    for (int size = 2; size <= n; size++) {
      for (int i = 0; i <= n - size; i++) {
        int j = i + size - 1;
        dp[i][j] = Math.max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1]);
      }
    }
    return dp[0][n - 1] > 0;
  }

  int[][][] dp;

  public boolean stoneGame(int[] piles) {
    int n = piles.length;
    if (n == 0) {
      return false;
    }
    dp = new int[n + 1][n + 1][2];
    for (int[][] arr : dp)
      for (int[] subArr : arr)
        Arrays.fill(subArr, -1);
    return stoneGameUtil(0, n, piles, 1) > 0;
  }

  private int stoneGameUtil(int start, int end, int[] piles, int isP1Turn) {
    if (start > end) {
      return 0;
    }

    if (start == end) {
      if (isP1Turn == 1) {
        return piles[start];
      } else {
        return -piles[start];
      }
    }

    if (dp[start][end][isP1Turn] != -1) {
      return dp[start][end][isP1Turn];
    }
    if (isP1Turn == 1) {
      dp[start][end][isP1Turn] = Math.max(piles[start] + stoneGameUtil(start + 1, end, piles, 0),
              piles[end] + stoneGameUtil(start, end - 1, piles, 0));
    } else {
      dp[start][end][isP1Turn] = Math.min(-piles[start] + stoneGameUtil(start + 1, end, piles, 1),
              -piles[end] + stoneGameUtil(start, end - 1, piles, 1));
    }
    return dp[start][end][isP1Turn];
  }

}
