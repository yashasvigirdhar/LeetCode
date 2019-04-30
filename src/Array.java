import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Array {

  public int[][] kClosest(int[][] points, int K) {
    PriorityQueue<double[]> pq = new PriorityQueue<>(new Comparator<double[]>() {
      @Override
      public int compare(double[] o1, double[] o2) {
        return Double.compare(o2[0], o1[0]);
      }
    });
    for (int i = 0; i < points.length; i++) {
      int x = Math.abs(points[i][0]);
      int y = Math.abs(points[i][1]);
      double d = Math.sqrt(x * x + y * y);
      if (pq.size() < K) {
        pq.add(new double[]{d, x, y});
      } else {
        if (d < pq.peek()[0]) {
          pq.poll();
          pq.add(new double[]{d, x, y});
        }
      }
    }
    int[][] res = new int[K][2];
    int i = 0;
    while (!pq.isEmpty()) {
      double[] poll = pq.poll();
      res[i++] = new int[]{(int) poll[1], (int) poll[2]};
    }
    return res;
  }

  HashMap<Integer, Boolean>[] partitionDp;

  public boolean canPartition(int[] nums) {
    partitionDp = new HashMap[nums.length];
    return canPartitionUtil(0, 0, nums);
  }

  private boolean canPartitionUtil(int idx, int curDiff, int[] nums) {
    if (idx == nums.length) {
      return curDiff == 0;
    }
    if (partitionDp[idx] != null && partitionDp[idx].containsKey(Math.abs(curDiff))) {
      return partitionDp[idx].get(curDiff);
    }
    if (partitionDp[idx] == null) {
      partitionDp[idx] = new HashMap<>();
    }
    if (canPartitionUtil(idx + 1, curDiff + nums[idx], nums)) {
      partitionDp[idx].put(curDiff, true);
      return true;
    }
    if (canPartitionUtil(idx + 1, curDiff - nums[idx], nums)) {
      partitionDp[idx].put(curDiff, true);
      return true;
    }
    partitionDp[idx].put(curDiff, false);
    return false;
  }

  public List<Integer> cellCompete(int[] states, int days) {
    int n = states.length;
    int[] newState = new int[n];
    Arrays.fill(newState, 0);
    int c = 0;
    while (c < days) {
      for (int i = 0; i < n; i++) {
        if (i == 0) {
          newState[i] = states[i + 1];
        } else if (i == n - 1) {
          newState[i] = states[i - 1];
        } else {
          newState[i] = states[i - 1] ^ states[i + 1];
        }
      }
      System.arraycopy(newState, 0, states, 0, n);
      Arrays.fill(newState, 0);
      c++;
    }
    List<Integer> res = new ArrayList<>();
    for (int state : states) {
      res.add(state);
    }
    return res;
  }

  public int combinationSum4(int[] nums, int target) {
    Arrays.sort(nums);
    int[] res = new int[1];
    backtrackForCombinationSum4(nums, target, res, 0);
    return res[0];
  }

  private void backtrackForCombinationSum4(int[] candidates, int target, int[] res, int curSum) {
    if (curSum == target) {
      res[0]++;
      return;
    }
    for (int i = 0; i < candidates.length; i++) {
      if (curSum + candidates[i] <= target) {
        backtrackForCombinationSum4(candidates, target, res, curSum + candidates[i]);
      } else {
        break;
      }
    }
  }

  public List<List<Integer>> combinationSum3(int k, int n) {
    int[] nums = new int[9];
    for (int i = 0; i < 9; i++) {
      nums[i] = i + 1;
    }
    ArrayList<List<Integer>> res = new ArrayList<>();
    backtrackForCombinationSum3(0, nums, n, k, res, new ArrayList<>(), 0);
    return res;
  }

  public void backtrackForCombinationSum3(int idx, int[] candidates, int target, int k, List<List<Integer>> res, List<Integer> cur, int curSum) {
    if (curSum == target) {
      res.add(new ArrayList<>(cur));
      return;
    }
    if (cur.size() == k) {
      return;
    }
    for (int i = idx; i < candidates.length; i++) {
      if (curSum + candidates[i] <= target) {
        cur.add(candidates[i]);
        backtrackForCombinationSum3(i + 1, candidates, target, k, res, cur, curSum + candidates[i]);
        cur.remove(cur.size() - 1);
      } else {
        break;
      }
    }
  }

  public List<List<Integer>> combinationSum2(int[] candidates, int target) {
    Arrays.sort(candidates);
    ArrayList<List<Integer>> res = new ArrayList<>();
    backtrackForCombinationSum2(0, candidates, target, res, new ArrayList<>(), 0);
    return res;
  }

  public void backtrackForCombinationSum2(int idx, int[] candidates, int target, List<List<Integer>> res, List<Integer> cur, int curSum) {
    if (curSum == target) {
      res.add(new ArrayList<>(cur));
      return;
    }
    for (int i = idx; i < candidates.length; i++) {
      if (i > idx && candidates[i] == candidates[i - 1]) continue;
      if (curSum + candidates[i] <= target) {
        cur.add(candidates[i]);
        backtrackForCombinationSum2(i + 1, candidates, target, res, cur, curSum + candidates[i]);
        cur.remove(cur.size() - 1);
      } else {
        break;
      }
    }
  }

  public List<List<Integer>> combinationSum(int[] candidates, int target) {
    Arrays.sort(candidates);
    ArrayList<List<Integer>> res = new ArrayList<>();
    backtrackForCombinationSum(0, candidates, target, res, new ArrayList<>(), 0);
    return res;
  }

  public void backtrackForCombinationSum(int idx, int[] candidates, int target, List<List<Integer>> res, List<Integer> cur, int curSum) {
    if (curSum == target) {
      res.add(new ArrayList<>(cur));
      return;
    }
    for (int i = idx; i < candidates.length; i++) {
      if (curSum + candidates[i] <= target) {
        cur.add(candidates[i]);
        backtrackForCombinationSum(i, candidates, target, res, cur, curSum + candidates[i]);
        cur.remove(cur.size() - 1);
      } else {
        break;
      }
    }
  }

  public List<List<Integer>> permuteUniqueUsingBacktrack(int[] nums) {
    ArrayList<List<Integer>> res = new ArrayList<>();
    backTrackForPermute(nums, res, new ArrayList<>());
    return res;
  }

  public void backTrackForPermuteUnique(int[] nums, List<List<Integer>> res, List<Integer> cur) {
    if (cur.size() == nums.length) {
      res.add(new ArrayList<>(cur));
    } else {
      for (int i = 0; i < nums.length; i++) {
        if (!cur.contains(nums[i])) {
          cur.add(nums[i]);
          backTrackForPermuteUnique(nums, res, cur);
          cur.remove((Integer) nums[i]);
        }
      }
    }
  }

  public List<List<Integer>> permuteUsingBacktrack(int[] nums) {
    ArrayList<List<Integer>> res = new ArrayList<>();
    backTrackForPermute(nums, res, new ArrayList<>());
    return res;
  }

  public void backTrackForPermute(int[] nums, List<List<Integer>> res, List<Integer> cur) {
    if (cur.size() == nums.length) {
      res.add(new ArrayList<>(cur));
    } else {
      for (int num : nums) {
        if (!cur.contains(num)) {
          cur.add(num);
          backTrackForPermute(nums, res, cur);
          cur.remove(cur.size() - 1);
        }
      }
    }
  }

  public List<List<Integer>> subsetsDupUsingBacktrack(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    backTrackForSubsets(0, nums, res, new ArrayList<>());
    return res;
  }

  public void backTrackForSubsetsDup(int idx, int[] nums, List<List<Integer>> res, List<Integer> cur) {
    res.add(new ArrayList<>(cur));
    for (int i = idx; i < nums.length; i++) {
      if (i > idx && nums[i] == nums[i - 1]) continue;
      cur.add(nums[i]);
      backTrackForSubsets(i + 1, nums, res, cur);
      cur.remove((Integer) nums[i]);
    }
  }

  public List<List<Integer>> subsetsUsingBacktrack(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    backTrackForSubsets(0, nums, res, new ArrayList<>());
    return res;
  }

  public void backTrackForSubsets(int idx, int[] nums, List<List<Integer>> res, List<Integer> cur) {
    res.add(new ArrayList<>(cur));
    for (int i = idx; i < nums.length; i++) {
      cur.add(nums[i]);
      backTrackForSubsets(i + 1, nums, res, cur);
      cur.remove((Integer) nums[i]);
    }
  }

  public List<List<Integer>> subsetsWithDup(int[] nums) {
    Set<Map<Integer, Integer>> set = new HashSet<>();
    int n = nums.length;
    int k = 0, c = 0;
    while (c < n) {
      k |= (1 << c);
      c++;
    }
    while (k >= 0) {
      Map<Integer, Integer> cur = new HashMap<>();
      for (int i = 0; i < nums.length; i++) {
        if ((k & (1 << i)) != 0) {
          cur.put(nums[i], cur.getOrDefault(nums[i], 0) + 1);
        }
      }
      set.add(cur);
      k--;
    }
    List<List<Integer>> res = new ArrayList<>();
    for (Map<Integer, Integer> map : set) {
      List<Integer> toAdd = new ArrayList<>();
      for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
        for (int i = 0; i < entry.getValue(); i++) {
          toAdd.add(entry.getKey());
        }
      }
      res.add(toAdd);
    }
    return res;
  }

  public List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> uniquePermutations = new ArrayList<>();
    uniquePermute(0, nums, uniquePermutations);
    return uniquePermutations;
  }

  private void uniquePermute(int idx, int[] nums, List<List<Integer>> uniquePermutations) {
    if (idx == nums.length) {
      List<Integer> cur = new ArrayList<>();
      for (int i = 0; i < nums.length; i++) {
        cur.add(nums[i]);
      }
      uniquePermutations.add(cur);
      return;
    }
    HashSet<Integer> start = new HashSet<>();
    for (int i = idx; i < nums.length; i++) {
      if (start.add(nums[i])) {
        swap(nums, idx, i);
        uniquePermute(idx + 1, nums, uniquePermutations);
        swap(nums, idx, i);
      }
    }
  }


  private List<List<Integer>> permutations;

  public List<List<Integer>> permute(int[] nums) {
    permutations = new ArrayList<>();
    permute(0, nums);
    return permutations;
  }

  private void permute(int idx, int[] nums) {
    if (idx == nums.length) {
      List<Integer> cur = new ArrayList<>();
      Arrays.stream(nums).forEach(cur::add);
      permutations.add(cur);
      return;
    }
    for (int i = idx; i < nums.length; i++) {
      swap(nums, idx, i);
      permute(idx + 1, nums);
      swap(nums, idx, i);
    }
  }

  public List<List<Integer>> subsets(int[] nums) {
    Set<List<Integer>> set = new HashSet<>();
    int n = nums.length;
    int k = 0, c = 0;
    while (c < n) {
      k |= (1 << c);
    }
    while (k >= 0) {
      List<Integer> cur = new ArrayList<>();
      for (int i = 0; i < nums.length; i++) {
        if ((k & (1 << i)) != 0) {
          cur.add(nums[i]);
        }
      }
      set.add(cur);
      k--;
    }
    return new ArrayList<>(set);
  }

  public int sumSubarrayMins(int[] a) {
    int mod = 1000000007;
    Stack<Integer> minStack = new Stack<>();
    int n = a.length;
    int[] leftMin = new int[n];
    for (int i = 0; i < n; i++) {
      while (!minStack.empty() && a[minStack.peek()] >= a[i]) {
        minStack.pop();
      }
      if (minStack.empty()) {
        leftMin[i] = -1;
      } else {
        leftMin[i] = minStack.peek();
      }
      minStack.push(i);

    }
    minStack.clear();
    int[] rightMin = new int[n];
    for (int i = n - 1; i >= 0; i--) {
      while (!minStack.empty() && a[minStack.peek()] >= a[i]) {
        minStack.pop();
      }
      if (minStack.empty()) {
        rightMin[i] = n;
      } else {
        rightMin[i] = minStack.peek();
      }
      minStack.push(i);

    }
    int ans = 0;
    for (int i = 0; i < n; i++) {
      int participation = rightMin[i] - i;
      int contribution = ((((i - leftMin[i]) * a[i]) % mod) * participation) % mod;
      ans = (ans + contribution) % mod;
    }
    return ans;
  }

  public List<List<Integer>> fourSumII(int[] nums, int target) {
    Arrays.sort(nums);
    int n = nums.length;
    Set<List<Integer>> res = new HashSet<>();
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        List<Integer> cur = new ArrayList<>();
        cur.add(nums[i]);
        cur.add(nums[j]);
        int curSum = nums[i] + nums[j];
        int toFind = target - curSum;
        int begin = j + 1, end = n - 1;
        while (begin < end) {
          if (nums[begin] + nums[end] == toFind) {
            List<Integer> list = new ArrayList<>(cur);
            list.add(nums[begin]);
            list.add(nums[end]);
            res.add(list);
            begin++;
            end--;
          } else if (nums[begin] + nums[end] < toFind) {
            begin++;
          } else {
            end--;
          }
        }
      }
    }
    return new ArrayList<>(res);
  }


  public List<List<Integer>> fourSum(int[] nums, int target) {
    Arrays.sort(nums);
    int n = nums.length;
    Set<List<Integer>> res = new HashSet<>();
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        int curSum = nums[i] + nums[j];
        int toFind = target - curSum;
        HashSet<Integer> map = new HashSet<>();
        for (int k = j + 1; k < n; k++) {
          int fourthRequired = toFind - nums[k];
          if (map.contains(fourthRequired)) {
            List<Integer> list = new ArrayList<>();
            list.add(nums[i]);
            list.add(nums[j]);
            list.add(fourthRequired);
            list.add(nums[k]);
            res.add(list);
          }
          map.add(nums[k]);
        }
      }
    }
    return new ArrayList<>(res);
  }

  public int[] smallestRange(List<List<Integer>> nums) {
    int k = nums.size();
    List<Pair<Integer, Integer>> res = IntStream.range(0, nums.size())
            .mapToObj(i -> nums.get(i).stream().map(x -> new Pair<>(x, i)))
            .flatMap(pair -> pair)
            .sorted(Comparator.comparingInt(o -> o.first)).collect(Collectors.toList());

    int r = 0, l = 0;
    Map<Integer, Integer> map = new HashMap<>();
    int[] ans = new int[2];
    int curD = Integer.MAX_VALUE;
    while (r < res.size()) {
      int typeToAdd = res.get(r).second;
      map.put(typeToAdd, map.getOrDefault(typeToAdd, 0) + 1);
      while (l <= r && map.size() == k) {
        int d = res.get(r).first - res.get(l).first + 1;
        if (d < curD || (d == curD && res.get(l).first < ans[0])) {
          curD = d;
          ans[0] = res.get(l).first;
          ans[1] = res.get(r).first;
        }
        int typeToRemove = res.get(l).second;
        map.put(typeToRemove, map.get(typeToRemove) - 1);
        if (map.get(typeToRemove) == 0) {
          map.remove(typeToRemove);
        }
        l++;
      }
      r++;
    }
    return ans;
  }

  public int largestSumAfterKNegationsII(int[] A, int K) {
    PriorityQueue<Integer> queue = new PriorityQueue<>(Comparator.reverseOrder());
    int sum = 0;
    for (int i = 0; i < A.length; i++) {
      sum += A[i];
      if (queue.size() == K) {
        if (queue.peek() > A[i]) {
          queue.poll();
          queue.add(A[i]);
        }
      } else {
        queue.add(A[i]);
      }
    }
    PriorityQueue<Integer> queue2 = new PriorityQueue<>();
    queue2.addAll(queue);
    while (K > 0) {
      int t = queue2.poll();
      sum -= t;
      t *= -1;
      sum += t;
      queue2.add(t);
      K--;
    }
    return sum;
  }

  public int largestSumAfterKNegations(int[] A, int K) {
    PriorityQueue<Integer> queue = new PriorityQueue<>();
    int sum = 0;
    for (int i = 0; i < A.length; i++) {
      sum += A[i];
      queue.add(A[i]);
    }
    while (K > 0) {
      int t = queue.poll();
      sum -= t;
      t *= -1;
      sum += t;
      queue.add(t);
      K--;
    }
    return sum;
  }

  public int subarraysDivByKOptimized(int[] A, int k) {
    int[] mods = new int[k];
    Arrays.fill(mods, 0);
    mods[0] = 1;
    int curSum = 0;
    int ans = 0;
    for (int aA : A) {
      int curMod = (curSum + aA % k) % k;
      ans += mods[curMod];
      mods[curMod]++;
    }
    return ans;
  }

  // this is also o(n^2).
  public int subarraysDivByKII(int[] A, int k) {
    int ans = 0;
    List<Integer> prev = new ArrayList<>();
    for (int curNum : A) {
      List<Integer> cur = new ArrayList<>();
      cur.add(curNum);
      if (curNum % k == 0) ans++;
      if (!prev.isEmpty()) {
        for (Integer prevSum : prev) {
          int curSum = prevSum + curNum;
          if (curSum % k == 0) {
            ans++;
          }
          cur.add(curSum);
        }
      }
      prev = cur;
    }
    return ans;
  }

  public int subarraysDivByK(int[] A, int k) {
    int ans = 0;
    for (int i = 0; i < A.length; i++) {
      int j = i;
      int curMod = 0;
      while (j >= 0) {
        curMod = (curMod + A[j] % k) % k;
        if (curMod == 0) {
          ans++;
        }
        j--;
      }
    }
    return ans;
  }

  public int maxSumTwoNoOverlap(int[] A, int L, int M) {
    int[] sum = new int[A.length];
    int csum = 0;
    for (int i = 0; i < A.length; i++) {
      csum += A[i];
      sum[i] = csum;
    }
    if (L < M) {
      int t = L;
      L = M;
      M = t;
    }
    int maxSum = 0;
    for (int end1 = L - 1; end1 < A.length; end1++) {
      int sum1 = sum[end1];
      int start1 = end1 - L + 1;
      if (start1 > 0) {
        sum1 -= sum[start1 - 1];
      }
      for (int end2 = M - 1; end2 < A.length; end2++) {
        int start2 = end2 - M + 1;
        if (!overlap(start1, end1, start2, end2)) {
          int sum2 = sum[end2];
          if (start2 > 0) {
            sum2 -= sum[start2 - 1];
          }
          maxSum = Math.max(maxSum, sum1 + sum2);
        }
      }
    }
    return maxSum;
  }

  private boolean overlap(int a, int b, int c, int d) {
    return Math.max(a, c) <= Math.min(b, d);
  }

  public List<Integer> findDuplicates(int[] nums) {
    List<Integer> ans = new ArrayList<>();
    for (int i = 0; i < nums.length; i++) {
      int idx = Math.abs(nums[i]);
      if (nums[idx - 1] < 0) {
        ans.add(idx);
      } else {
        nums[idx - 1] *= -1;
      }
    }
    return ans;
  }

  public int numSubarrayBoundedMaxOptimized(int[] A, int L, int R) {
    int ans = 0, last = 0, res = 0;
    for (int i = 0; i < A.length; i++) {
      if (A[i] >= L && A[i] <= R) {
        res = (i - last + 1);
        ans += res;
      } else if (A[i] < L) {
        ans += res;
      } else {
        res = 0;
        last = i + 1;
      }
    }
    return ans;
  }

  public int numSubarrayBoundedMax(int[] A, int L, int R) {
    int ans = 0;
    for (int i = 0; i < A.length; i++) {
      int max = 0;
      for (int j = i; j < A.length; j++) {
        max = Math.max(max, A[j]);
        if (max >= L && max <= R) {
          ans++;
        } else {
          break;
        }
      }
    }
    return ans;
  }

  public void gameOfLifeMoreMemoryOptimized(int[][] board) {
    int m = board.length;
    if (m == 0) return;
    int n = board[0].length;
    int[][] dims = new int[][]{
            {-1, 0},
            {-1, 1},
            {0, 1},
            {1, 1},
            {1, 0},
            {1, -1},
            {0, -1},
            {-1, -1}
    };

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        int liveNeighbours = 0;
        for (int[] dim : dims) {
          int newI = i + dim[0];
          int newJ = j + dim[1];
          boolean isValid = newI >= 0 && newJ >= 0 && newI < m && newJ < n;
          if (isValid && (board[newI][newJ] & 1) != 0) {
            liveNeighbours++;
          }
        }
        if (board[i][j] == 0 && liveNeighbours == 3) {
          board[i][j] = 2;
        } else if (board[i][j] == 1 && liveNeighbours >= 2 && liveNeighbours <= 3) {
          board[i][j] = 3;
        }
      }
    }

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        board[i][j] >>= 1;
      }
    }

  }

  public void gameOfLifeMemoryOptimized(int[][] board) {
    int m = board.length;
    if (m == 0) return;
    int n = board[0].length;
    int[][] dims = new int[][]{
            {-1, 0},
            {-1, 1},
            {0, 1},
            {1, 1},
            {1, 0},
            {1, -1},
            {0, -1},
            {-1, -1}
    };
    int[] result = new int[n];

    for (int i = 0; i < m; i++) {
      int[] temp = new int[n];
      for (int j = 0; j < n; j++) {
        int liveNeighbours = 0;
        for (int[] dim : dims) {
          int newI = i + dim[0];
          int newJ = j + dim[1];
          boolean isValid = newI >= 0 && newJ >= 0 && newI < m && newJ < n;
          if (isValid && board[newI][newJ] == 1) {
            liveNeighbours++;
          }
        }
        temp[j] = board[i][j];
        if (board[i][j] == 0) {
          if (liveNeighbours == 3) {
            temp[j] = 1;
          }
        } else {
          if (liveNeighbours < 2 || liveNeighbours > 3) {
            temp[j] = 0;
          }
        }
      }
      if (i > 0) {
        System.arraycopy(result, 0, board[i - 1], 0, n);
      }
      System.arraycopy(temp, 0, result, 0, n);
    }
    System.arraycopy(result, 0, board[m - 1], 0, n);
  }

  public void gameOfLife(int[][] board) {
    int m = board.length;
    if (m == 0) return;
    int n = board[0].length;
    int[][] dims = new int[][]{
            {-1, 0},
            {-1, 1},
            {0, 1},
            {1, 1},
            {1, 0},
            {1, -1},
            {0, -1},
            {-1, -1}
    };
    int[][] result = new int[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        int liveNeighbours = 0;
        for (int[] dim : dims) {
          int newI = i + dim[0];
          int newJ = j + dim[1];
          boolean isValid = newI >= 0 && newJ >= 0 && newI < m && newJ < n;
          if (isValid && board[newI][newJ] == 1) {
            liveNeighbours++;
          }
        }
        result[i][j] = board[i][j];
        if (board[i][j] == 0) {
          if (liveNeighbours == 3) {
            result[i][j] = 1;
          }
        } else {
          if (liveNeighbours < 2 || liveNeighbours > 3) {
            result[i][j] = 0;
          }
        }
      }
    }
    for (int i = 0; i < m; i++) {
      System.arraycopy(result[i], 0, board[i], 0, n);
    }
  }

  public int maxSubArray(int[] nums) {
    int curSum = 0, maxSum = Integer.MIN_VALUE;
    for (int num : nums) {
      curSum += num;
      maxSum = Math.max(curSum, maxSum);
      if (curSum < 0) {
        curSum = 0;
      }
    }
    return maxSum;
  }

  public int maxProfitII(int[] prices, int fee) {
    int i = 0;
    int profit = 0;
    while (i < prices.length) {
      while (i < prices.length - 1 && prices[i] >= prices[i + 1]) {
        i++;
      }
      if (i == prices.length - 1) break;
      int buyPrice = prices[i];
      int sellPrice = prices[i];
      i++;
      while (i < prices.length) {
        if (prices[i] < sellPrice) {
          if (sellPrice - prices[i] > fee) {
            break;
          } else {
            i++;
          }
        } else {
          sellPrice = prices[i];
          i++;
        }
      }

      int curProfit = sellPrice - buyPrice - fee;
      if (curProfit > 0) {
        profit += curProfit;
      }
    }
    return profit > 0 ? profit : 0;
  }

  public int maxProfitII(int[] prices) {
    int i = 0;
    int profit = 0;
    while (i < prices.length) {
      while (i < prices.length - 1 && prices[i] >= prices[i + 1]) {
        i++;
      }
      if (i == prices.length - 1) break;
      int buyPrice = prices[i];
      i++;
      int sellPrice = prices[i];
      i++;
      while (i < prices.length && prices[i] >= sellPrice) {
        sellPrice = prices[i];
        i++;
      }
      profit += (sellPrice - buyPrice);
    }
    return profit;
  }

  public int maxProfit(int[] prices) {
    if (prices.length == 0) return 0;
    int min = Integer.MAX_VALUE;
    int profit = 0;
    for (int price : prices) {
      min = Math.min(min, price);
      profit = Math.max(profit, price - min);
    }
    return profit;
  }


  public void wiggleSortII(int[] nums) {
    Arrays.sort(nums);
    int l = nums.length;
    int[] t = new int[l];
    int idx = 0;
    int i = (l + 1) / 2 - 1, j = l - 1;
    while (j >= (l + 1) / 2) {
      t[idx++] = nums[i--];
      t[idx++] = nums[j--];
    }
    if (i > 0) {
      t[idx] = nums[i];
    }
    System.arraycopy(t, 0, nums, 0, l);
  }

  public void wiggleSortI_II(int[] nums) {
    int idx = 0;
    boolean small = false;
    while (idx < nums.length - 1) {
      if (small) {
        if (nums[idx + 1] > nums[idx]) {
          int t = nums[idx + 1];
          nums[idx + 1] = nums[idx];
          nums[idx] = t;
        }
      } else {
        if (nums[idx + 1] < nums[idx]) {
          int t = nums[idx + 1];
          nums[idx + 1] = nums[idx];
          nums[idx] = t;
        }
      }
      idx++;
      small = !small;
    }
  }

  public void wiggleSort(int[] nums) {
    int idx = 0;
    boolean small = true;
    while (idx < nums.length) {
      int jdx;
      if (small) {
        int min = nums[idx], minI = idx;
        for (int i = idx + 1; i < nums.length; i++) {
          if (nums[i] < min) {
            min = nums[i];
            minI = i;
          }
        }
        jdx = minI;
      } else {
        int max = nums[idx], maxI = idx;
        for (int i = idx + 1; i < nums.length; i++) {
          if (nums[i] > max) {
            max = nums[i];
            maxI = i;
          }
        }
        jdx = maxI;
      }
      int temp = nums[idx];
      nums[idx] = nums[jdx];
      nums[jdx] = temp;
      small = !small;
      idx++;
    }
  }

  public int longestMountainIII(int[] A) {
    int n = A.length;
    int[] l = new int[n], r = new int[n];
    Arrays.fill(l, 0);
    Arrays.fill(r, 0);
    for (int i = 1; i < n; i++) {
      if (A[i] > A[i - 1]) {
        l[i] = l[i - 1] + 1;
      }
    }
    for (int i = n - 2; i >= 0; i--) {
      if (A[i] > A[i + 1]) {
        r[i] = r[i] + 1;
      }
    }
    int ans = 0;
    for (int i = 1; i < n - 1; i++) {
      ans = Math.max(ans, l[i] + r[i] + 1);
    }
    return ans;
  }

  public int longestMountainII(int[] A) {
    int start = 0;
    int n = A.length;
    int ans = 0;
    while (start < n) {
      int end = start;
      if (end + 1 < n && A[end] < A[end + 1]) {
        while (end + 1 < n && A[end] < A[end + 1]) {
          end++;
        }
        if (end + 1 < n && A[end] > A[end + 1]) {
          while (end + 1 < n && A[end] > A[end + 1]) {
            end++;
          }
          ans = Math.max(ans, end - start + 1);
        }
      }
      start = Math.max(end, start + 1);
    }
    return ans;
  }

  public int longestMountain(int[] A) {
    int maxLen = 0;
    if (A.length < 3) {
      return maxLen;
    }
    for (int i = 1; i < A.length - 1; i++) {
      int j = i - 1;
      int top = A[i];
      //left side
      while (j >= 0 && A[j] < top) {
        top = A[j];
        j--;
      }
      int start = j + 1;
      //right side
      j = i + 1;
      top = A[i];
      while (j < A.length && A[j] < top) {
        top = A[j];
        j++;
      }
      int end = j - 1;
      int curLen = (start != i && end != i) ? end - start + 1 : 0;
      if (curLen >= 3) {
        maxLen = Math.max(maxLen, curLen);
      }
    }
    return maxLen;
  }

  public int totalFruit(int[] tree) {
    int l = 0, r = 0;
    Map<Integer, Integer> map = new HashMap<>();
    int ans = 0;
    while (r < tree.length) {
      map.put(tree[r], map.getOrDefault(tree[r], 0) + 1);
      while (l <= r && map.size() > 2) {
        map.put(tree[l], map.get(tree[l]) - 1);
        if (map.get(tree[l]) == 0) {
          map.remove(tree[l]);
        }
        l++;
      }
      ans = Math.max(ans, r - l + 1);
      r++;
    }
    return ans;
  }

  public int subarraySumII(int[] nums, int k) {
    Map<Integer, Integer> map = new HashMap<>();
    int sum = 0, count = 0;
    map.put(0, 1);
    for (int num : nums) {
      sum += num;
      count += map.getOrDefault(sum - k, 0);
      map.put(sum, map.getOrDefault(sum, 0) + 1);
    }
    return count;
  }

  public void moveZeroes(int[] nums) {
    int left = -1, right = 0;
    while (right < nums.length) {
      if (nums[right] != 0) {
        left++;
        if (left != right) {
          int temp = nums[left];
          nums[left] = nums[right];
          nums[right] = temp;
        }
      }
      right++;
    }
  }

  public String findReplaceString(String s, int[] indexes, String[] sources, String[] targets) {
    Map<Integer, Pair<String, String>> map = new HashMap<>();
    for (int i = 0; i < indexes.length; i++) {
      map.put(indexes[i], new Pair<>(sources[i], targets[i]));
    }
    int idx = 0;
    StringBuilder ans = new StringBuilder();
    while (idx < s.length()) {
      if (map.containsKey(idx)) {
        Pair<String, String> curPair = map.get(idx);
        String source = curPair.first;
        if (idx + source.length() - 1 < s.length() && s.substring(idx, idx + source.length()).equals(source)) {
          idx += source.length();
          ans.append(curPair.second);
        } else {
          ans.append(s.charAt(idx));
          idx++;
        }
      } else {
        ans.append(s.charAt(idx));
        idx++;
      }
    }
    return ans.toString();
  }

  public int[] plusOne(int[] digits) {
    int carry = 1;
    ArrayList<Integer> result = new ArrayList<>(digits.length + 1);
    int sum;
    for (int i = digits.length - 1; i >= 0; i--) {
      sum = digits[i] + carry;
      result.add(sum % 10);
      carry = sum / 10;
    }
    if (carry != 0) {
      result.add(carry);
    }
    Collections.reverse(result);
    return result.stream().mapToInt(value -> value).toArray();
  }

  public int[][] multiplyIII(int[][] A, int[][] B) {
    int a = A.length;
    int b = A[0].length;
    int c = B[0].length;
    int[][] result = new int[a][c];
    for (int[] arr : result) Arrays.fill(arr, 0);
    for (int i = 0; i < a; i++) {
      for (int k = 0; k < b; k++) {
        if (A[i][k] != 0) {
          for (int j = 0; j < c; j++) {
            result[i][j] += A[i][k] * B[k][j];
          }
        }
      }
    }
    return result;
  }

  public int[][] multiplyII(int[][] A, int[][] B) {
    int a = A.length;
    int b = A[0].length;
    int c = B[0].length;
    int[][] result = new int[a][c];
    Map<Integer, List<Integer>> rowsMap = new HashMap<>();
    Map<Integer, List<Integer>> colsMap = new HashMap<>();
    for (int row = 0; row < a; row++) {
      for (int col = 0; col < c; col++) {
        boolean rowAlreadyComputed = false, colAlreadyComputed = false;
        if (rowsMap.containsKey(row)) {
          rowAlreadyComputed = true;
        }
        if (colsMap.containsKey(col)) {
          colAlreadyComputed = true;
        }
        if (rowAlreadyComputed && colAlreadyComputed) {
          result[row][col] = multiplyUtil(A, B, rowsMap.get(row), colsMap.get(col), row, col);
          continue;
        }
        if (!rowAlreadyComputed) {
          rowsMap.put(row, new ArrayList<>());
        }
        if (!colAlreadyComputed) {
          colsMap.put(col, new ArrayList<>());
        }
        int sum = 0;
        for (int i = 0; i < b; i++) {
          sum += A[row][i] * B[i][col];
          if (A[row][i] != 0 && !rowAlreadyComputed) {
            rowsMap.get(row).add(i);
          }
          if (B[i][col] != 0 && !colAlreadyComputed) {
            colsMap.get(col).add(i);
          }
        }
        result[row][col] = sum;
      }
    }
    return result;
  }

  private int multiplyUtil(int[][] A, int[][] B, List<Integer> rowIdx, List<Integer> colIdx, int row, int col) {
    int i = 0, j = 0, sum = 0;
    while (i < rowIdx.size() && j < colIdx.size()) {
      if (rowIdx.get(i).equals(colIdx.get(j))) {
        sum += A[row][rowIdx.get(i)] * B[col][colIdx.get(j)];
        i++;
        j++;
      } else if (rowIdx.get(i) < colIdx.get(j)) {
        i++;
      } else {
        j++;
      }
    }
    return sum;
  }

  public int[][] multiply(int[][] A, int[][] B) {
    int a = A.length;
    // if(a==0)return new int[0][0];
    int b = A[0].length;
    int c = B[0].length;
    int[][] result = new int[a][c];
    for (int row = 0; row < a; row++) {
      for (int col = 0; col < c; col++) {
        int sum = 0;
        for (int i = 0; i < b; i++) {
          sum += A[row][i] * B[i][col];
        }
        result[row][col] = sum;
      }
    }
    return result;
  }

  public List<String> summaryRanges(int[] nums) {
    List<String> ans = new ArrayList<>();
    int n = nums.length;
    if (n == 0) return ans;
    int start = nums[0], end = nums[0];
    boolean rangeInProgress = true;
    for (int i = 1; i < n; i++) {
      if (rangeInProgress) {
        if (nums[i] == end + 1) {
          end = nums[i];
        } else {
          StringBuilder builder = new StringBuilder().append(start);
          if (end != start) {
            builder.append("->").append(end);
          }
          ans.add(builder.toString());
          rangeInProgress = false;
        }
      }
      if (!rangeInProgress) {
        start = nums[i];
        end = nums[i];
        rangeInProgress = true;
      }
    }
    if (rangeInProgress) {
      StringBuilder builder = new StringBuilder().append(start);
      if (end != start) {
        builder.append("->").append(end);
      }
      ans.add(builder.toString());
    }
    return ans;
  }

  public int findKthLargestIII(int[] nums, int k) {
    shuffle(nums);
    int l = 0, r = nums.length - 1;
    while (l <= r) {
      if (l == r) {
        return nums[l];
      }
      int curIdx = partition(nums, l, r);
      if (curIdx == nums.length - k) {
        return nums[curIdx];
      } else if (nums.length - k < curIdx) {
        r = curIdx - 1;
      } else {
        l = curIdx + 1;
      }
    }
    return -1;
  }

  private int partition(int[] nums, int left, int right) {
    int pivot = nums[right];
    int pIdx = left;
    for (int i = left; i < right; i++) {
      if (nums[i] <= pivot) {
        swap(nums, pIdx, i);
        pIdx++;
      }
    }
    swap(nums, pIdx, right);
    return pIdx;
  }

  private void shuffle(int[] nums) {
    int count = 0;
    int n = nums.length;
    while (n > 0) {
      int idx = new Random().nextInt(n);
      swap(nums, idx, n - 1);
      n--;
    }
  }

  private void swap(int[] nums, int left, int right) {
    int temp = nums[right];
    nums[right] = nums[left];
    nums[left] = temp;
  }

  public int findKthLargestII(int[] nums, int k) {
    PriorityQueue<Integer> queue = new PriorityQueue<>();
    for (int n : nums) {
      if (queue.size() < k) {
        queue.add(n);
      } else if (queue.peek() <= n) {
        queue.poll();
        queue.add(n);
      }
    }
    return queue.poll();
  }

  public int findKthLargest(int[] nums, int k) {
    PriorityQueue<Integer> queue = new PriorityQueue<>((o1, o2) -> o2 - o1);
    for (int n : nums) {
      queue.add(n);
    }
    int ans = 0;
    while (k > 0) {
      ans = queue.poll();
      k--;
    }
    return ans;
  }

  public List<Integer> topKFrequentII(int[] nums, int k) {
    HashMap<Integer, Integer> map = new HashMap<>();
    for (int num : nums) {
      map.put(num, map.getOrDefault(num, 0) + 1);
    }
    List<Integer>[] buckets = new List[nums.length + 1];
    for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
      if (buckets[entry.getValue()] == null) {
        buckets[entry.getValue()] = new ArrayList<>();
      }
      buckets[entry.getValue()].add(entry.getKey());
    }
    List<Integer> ans = new ArrayList<>();
    for (int i = buckets.length - 1; i >= 0 && k > 0; i--) {
      if (buckets[i] != null) {
        ans.addAll(buckets[i].subList(0, k));
        k -= buckets[i].size();
      }
    }
    return ans;
  }

  public List<Integer> topKFrequent(int[] nums, int k) {
    HashMap<Integer, Integer> map = new HashMap<>();
    for (int num : nums) {
      map.put(num, map.getOrDefault(num, 0) + 1);
    }
    PriorityQueue<Map.Entry<Integer, Integer>> queue = new PriorityQueue<>(Comparator.comparing(Map.Entry::getValue));
    int finalK = k;
    map.entrySet().forEach(pair -> {
      if (queue.size() < finalK) {
        queue.add(pair);
      } else if (queue.peek().getValue() < pair.getValue()) {
        queue.poll();
        queue.add(pair);
      }
    });
    List<Integer> ans = new ArrayList<>();
    while (k > 0 && !queue.isEmpty()) {
      ans.add(queue.poll().getKey());
      k--;
    }
    Collections.reverse(ans);
    return ans;
  }

  public int maximumProduct(int[] nums) {
    Arrays.sort(nums);
    int n = nums.length;
    int c1 = nums[n - 1] + nums[n - 2] + nums[n - 3];
    int c2 = nums[0] * nums[1] * nums[n - 1];
    return Math.max(c1, c2);
  }

  public int arrayPairSum(int[] nums) {
    if (nums.length == 0) return 0;
    Arrays.sort(nums);
    int idx = 0, sum = 0;
    while (idx < nums.length - 1) {
      sum += nums[idx];
      idx += 2;
    }
    return sum;
  }

  public List<Integer> getRowII(int numRows) {
    List<Integer> ans = new ArrayList<>();
    int[] cur = new int[numRows + 1];
    Arrays.fill(cur, 0);
    cur[0] = 1;
    int n = 1;
    while (n <= numRows) {
      for (int j = n; j >= 1; j--) {
        cur[j] += cur[j - 1];
      }
      n++;
    }
    for (int i : cur) {
      ans.add(i);
    }
    return ans;
  }

  public List<Integer> getRow(int numRows) {
    List<Integer> ans = new ArrayList<>();
    int[] cur = new int[numRows + 1];
    Arrays.fill(cur, -1);
    cur[0] = 1;
    int n = 1;
    while (n <= numRows) {
      int prev = -1;
      int idx = 0;
      cur[idx++] = 1;
      int l = 0, r = 1;
      while (cur[r] != -1) {
        int temp = cur[l] + cur[r];
        if (prev != -1) {
          cur[idx++] = prev;
        }
        prev = temp;
        l++;
        r++;
      }
      if (prev != -1) {
        cur[idx++] = prev;
      }
      cur[idx] = 1;
      n++;
    }
    for (int i : cur) {
      ans.add(i);
    }
    return ans;
  }

  public List<List<Integer>> generate(int numRows) {
    List<List<Integer>> ans = new ArrayList<>();
    if (numRows == 0) return ans;
    List<Integer> prev = new ArrayList<>();
    prev.add(1);
    ans.add(prev);
    int n = 2;
    while (n <= numRows) {
      List<Integer> cur = new ArrayList<>();
      cur.add(1);
      int l = 0, r = 1;
      while (r < prev.size()) {
        cur.add(prev.get(l) + prev.get(r));
        l++;
        r++;
      }
      cur.add(1);
      ans.add(cur);
      prev = cur;
      n++;
    }
    return ans;
  }

  public int[][] flipAndInvertImage(int[][] A) {
    for (int i = 0; i < A.length; i++) {
      for (int j = 0, k = A[i].length - 1; j < k; j++, k--) {
        int temp = A[i][k];
        A[i][k] = A[i][j] ^ 1;
        A[i][j] = temp ^ 1;
      }
    }
    return A;
  }

  public List<Integer> pancakeSort(int[] a) {
    int size = a.length;
    List<Integer> ans = new ArrayList<>();
    while (size > 1) {
      ans.addAll(pancakeSort(a, size));
      size--;
    }
    return ans;
  }

  public List<Integer> pancakeSort(int[] a, int n) {
    List<Integer> ans = new ArrayList<>();
    if (n == 0) {
      return ans;
    }
    int mx = a[0], mxIdx = 0;
    for (int i = 1; i < n; i++) {
      if (a[i] >= mx) {
        mx = a[i];
        mxIdx = i;
      }
    }
    if (mxIdx == n - 1) {
      return ans;
    }

    int i = 0, j = mxIdx;
    if (i != mxIdx) {
      ans.add(mxIdx);
      reverse(a, i, j);
    }

    ans.add(n - 1);
    i = 0;
    j = n - 1;
    reverse(a, i, j);
    return ans;
  }

  private void reverse(int[] a, int i, int j) {
    while (i < j) {
      int temp = a[i];
      a[i] = a[j];
      a[j] = temp;
      i++;
      j--;
    }
  }


  public int majorityElementBitwise(int[] nums) {
    int ans = 0;
    for (int i = 0, mask = 1; i < 32; i++, mask <<= 1) {
      int bitCount = 0;
      for (int num : nums) {
        if ((num & mask) != 0) {
          bitCount++;
          if (bitCount > nums.length / 2) {
            ans |= mask;
            break;
          }
        }
      }
    }
    return ans;
  }

  public List<Integer> majorityElementMedium(int[] nums) {
    int cand1 = -1, cand2 = -1, cnt1 = 0, cnt2 = 0;
    for (int num : nums) {
      if (num == cand1) {
        cnt1++;
      } else if (num == cand2) {
        cnt2++;
      } else if (cnt1 == 0) {
        cand1 = num;
        cnt1++;
      } else if (cnt2 == 0) {
        cand2 = num;
        cnt2++;
      } else {
        cnt1--;
        cnt2--;
      }
    }
    cnt1 = 0;
    cnt2 = 0;
    List<Integer> ans = new ArrayList<>();
    for (int num : nums) {
      if (num == cand1) cnt1++;
      if (num == cand2) cnt2++;
    }
    if (cnt1 > nums.length / 3) {
      ans.add(cand1);
    }
    if (cnt2 > nums.length / 3) {
      ans.add(cand2);
    }
    return ans;
  }

  public int majorityElement(int[] nums) {
    int count = 0, maj = -1;
    for (int num : nums) {
      if (count == 0) {
        maj = num;
        count++;
      } else if (num == maj) {
        count++;
      } else {
        count--;
      }
    }
    return maj;
  }

  public int findShortestSubArrayII(int[] nums) {
    Map<Integer, Integer> count = new HashMap<>(), first = new HashMap<>();
    int n = nums.length;
    int degree = 1, ans = n;
    for (int i = 0; i < n; i++) {

      first.putIfAbsent(nums[i], i);

      int curCount = count.getOrDefault(nums[i], 0) + 1;
      if (curCount == degree) {
        ans = Math.min(ans, i - first.get(nums[i]) + 1);
      } else if (curCount > degree) {
        degree = curCount;
        ans = i - first.get(nums[i]) + 1;
      }
      count.put(nums[i], curCount);
    }
    return ans;
  }

  public int findShortestSubArray(int[] nums) {
    int n = nums.length;
    Map<Integer, Integer> map = new HashMap<>();
    int degree = 1;
    for (int num : nums) {
      int value = map.getOrDefault(num, 0) + 1;
      map.put(num, value);
      degree = Math.max(degree, value);
    }
    map.clear();
    int l = 0, r = 0, ans = n;
    while (r < n) {
      int key = nums[r];
      int value = map.getOrDefault(key, 0) + 1;
      map.put(key, value);
      while (l <= r && value >= degree) {
        ans = Math.min(ans, r - l + 1);
        map.put(nums[l], map.get(nums[l]) - 1);
        if (nums[l] == key) {
          value--;
        }
        l++;
      }
      r++;
    }
    return ans;
  }

  public int maxWidthRampIII(int[] A) {
    int n = A.length;
    Stack<Integer> st = new Stack<>();
    for (int i = 0; i < n; i++) {
      if (st.empty() || A[i] < A[st.peek()]) {
        st.push(i);
      }
    }
    int ans = 0;
    for (int i = n - 1; i >= 0; i--) {
      while (!st.empty() && A[i] > A[st.peek()]) {
        ans = Math.max(i - st.pop(), ans);
      }
    }
    return ans;
  }

  public int maxWidthRampII(int[] A) {
    int n = A.length;
    int[] rMax = new int[n];
    rMax[n - 1] = A[n - 1];
    for (int i = n - 2; i >= 0; i--) {
      rMax[i] = Math.max(rMax[i + 1], A[i]);
    }
    int left = 0, right = 0;
    int ans = 0;
    while (right < n) {
      while (left < right && A[left] > rMax[right]) {
        left++;
      }
      ans = Math.max(ans, right - left);
      right++;
    }
    return ans;
  }

  public List<String> findMissingRangesIII(int[] nums, int lower, int upper) {
    int next = lower;
    List<String> result = new ArrayList<>();
    for (int num : nums) {
      if (num != next) {
        StringBuilder builder = new StringBuilder();
        builder.append(next);
        int end = (num - 1);
        if (next != end) {
          builder.append("->");
          builder.append(end);
        }
        result.add(builder.toString());
      }
      if (num == upper) {
        return result;
      }
      next = num + 1;
    }
    if (next <= upper) {
      StringBuilder builder = new StringBuilder();
      builder.append(next);
      if (next != upper) {
        builder.append("->");
        builder.append(upper);
      }
      result.add(builder.toString());
    }
    return result;
  }

  public List<String> findMissingRangesII(int[] nums, int lower, int upper) {
    List<String> result = new ArrayList<>();
    long ctr = lower;
    int idx = 0;
    int curStart = -1;
    while (ctr <= upper && idx < nums.length) {
      if (nums[idx] == ctr) {
        if (curStart != -1) {
          StringBuilder builder = new StringBuilder();
          builder.append(curStart);
          int end = (int) (ctr - 1);
          if (curStart != end) {
            builder.append("->");
            builder.append(end);
          }
          result.add(builder.toString());
          curStart = -1;
        }
        idx++;
      } else {
        if (curStart == -1) {
          curStart = (int) ctr;
        }
      }
      ctr++;
    }
    if (ctr <= upper) {
      int start;
      if (curStart != -1) {
        start = curStart;
      } else {
        start = (int) ctr;
      }
      StringBuilder builder = new StringBuilder();
      builder.append(start);
      if (start != upper) {
        builder.append("->");
        builder.append(upper);
      }
      result.add(builder.toString());
    }
    return result;
  }

  public List<String> findMissingRanges(int[] nums, int lower, int upper) {
    int n = nums.length;
    List<String> ans = new ArrayList<>();
    long next = lower;
    for (int i = 0; i < n; i++) {
      if (nums[i] < next) {
        continue;
      }

      if (nums[i] == next) {
        next++;
        continue;
      }
      StringBuilder b = new StringBuilder(String.valueOf(next));
      if (next != nums[i] - 1) {
        b.append("->").append(nums[i] - 1);
      }
      ans.add(b.toString());
      next = (long) nums[i] + 1;
    }
    if (next <= upper) {
      StringBuilder b = new StringBuilder(String.valueOf(next));
      if (next < upper) {
        b.append("->").append(upper);
      }
      ans.add(b.toString());
    }
    return ans;
  }

  public int maxWidthRamp(int[] A) {
    int n = A.length;
    Integer[] b = new Integer[n];
    for (int i = 0; i < n; i++) {
      b[i] = i;
    }
    Arrays.sort(b, Comparator.comparingInt(i -> A[i]));
    int mn = n;
    int ans = 0;
    for (int i = 0; i < n; i++) {
      ans = Math.max(ans, b[i] - mn);
      mn = Math.min(mn, b[i]);
    }
    return ans;
  }

  public boolean canReorderDoubled(int[] A) {
    Map<Integer, Integer> map = new HashMap<>();
    Arrays.sort(A);
    for (int aA : A) {
      int halfKey = aA / 2;
      int doubleKey = aA * 2;
      int searchKey;
      if (aA > 0) {
        searchKey = halfKey;
      } else {
        searchKey = doubleKey;
      }
      if (((searchKey == doubleKey) || (searchKey == halfKey && Math.floorMod(aA, 2) == 0)) && map.containsKey(searchKey)) {
        map.put(searchKey, map.get(searchKey) - 1);
        if (map.get(searchKey) == 0) {
          map.remove(searchKey);
        }
      } else {
        map.put(aA, map.getOrDefault(aA, 0) + 1);
      }
    }
    return map.size() == 0;
  }


  public boolean circularArrayLoop(int[] nums) {
    int n = nums.length;
    int[] cycle = new int[n];
    boolean[] inStack = new boolean[n];
    for (int i = 0; i < n; i++) {
      cycle[i] = -1;
      inStack[i] = false;
    }
    for (int i = 0; i < nums.length; i++) {
      if (cycle[i] == -1) {
        // unvisited till now
        if (dfs(i, nums, cycle, 0, inStack)) {
          return true;
        }
      }
    }
    return false;
  }

  private boolean dfs(int i, int[] nums, int[] cycle, int curStackSize, boolean[] inStack) {
    if (inStack[i]) {
      return curStackSize > 1;
    }

    if (cycle[i] == -1) {
      inStack[i] = true;
      curStackSize++;
      int nextPos = Math.floorMod(i + nums[i], nums.length);
      boolean b;
      if (i != nextPos && (nums[i] * nums[nextPos] > 0)) {
        b = dfs(nextPos, nums, cycle, curStackSize, inStack);
      } else {
        b = false;
      }
      cycle[i] = b ? 1 : 0;
      inStack[i] = false;
    }

    return cycle[i] == 1;
  }

  public int numPairsDivisibleBy60(int[] time) {
    int ans = 0, k = 60;
    int[] m = new int[60];
    for (int i = 0; i < 60; i++) {
      m[i] = 0;
    }

    for (int aTime : time) {
      int t = aTime % k;
      int toFind = (k - t) % k;
      ans += m[toFind];
      m[t] += 1;
    }

    return ans;
  }

  public int removeDuplicatesII_2(int[] nums) {
    int l = 0, r = 1, count = 1;
    while (r < nums.length) {
      if (nums[r] != nums[r - 1]) {
        count = 1;
        nums[++l] = nums[r];
      } else {
        if (count < 2) {
          count++;
          nums[++l] = nums[r];
        }
      }
      r++;
    }
    return l + 1;
  }

  public int removeDuplicatesII(int[] nums) {
    Map<Integer, Integer> map = new HashMap<>();
    int l = 0, r = 0;
    while (r < nums.length) {
      if (nums[r] != nums[l] || map.getOrDefault(nums[r], 0) < 2) {
        l++;
        int t = nums[r];
        nums[r] = nums[l];
        nums[l] = t;
      }
      map.put(nums[r], map.getOrDefault(nums[r], 0));
      r++;
    }
    return l + 1;
  }

  public int[] intersect(int[] nums1, int[] nums2) {
    ArrayList<Integer> ans = new ArrayList<>();
    Map<Integer, Integer> map = new HashMap<>();
    for (int aNums1 : nums1) {
      map.put(aNums1, map.getOrDefault(aNums1, 0) + 1);
    }
    for (int aNums2 : nums2) {
      if (map.getOrDefault(aNums2, 0) > 0) {
        ans.add(aNums2);
        map.put(aNums2, map.get(aNums2) - 1);
      }
    }
    return ans.stream().mapToInt(value -> value).toArray();
  }

  public int removeDuplicates(int[] nums) {
    int l = 0, r = 0;
    while (r < nums.length) {
      if (nums[r] != nums[l]) {
        l++;
        int t = nums[r];
        nums[r] = nums[l];
        nums[l] = t;
      }
      r++;
    }
    return l + 1;
  }

  public int maxSubArrayLen(int[] nums, int k) {
    Map<Integer, Integer> m = new HashMap<>();
    int cSum = 0;
    int maxAns = 0;
    m.put(0, -1);
    for (int i = 0; i < nums.length; i++) {
      cSum += nums[i];
      if (cSum >= k) {
        int diff = cSum - k;
        if (m.containsKey(diff)) {
          maxAns = Math.max(maxAns, i - m.get(diff));
        }
      }
    }
    return maxAns;
  }


  public int subarraySum(int[] nums, int k) {
    HashMap<Integer, Boolean> map = new HashMap<>();
    map.put(0, true);
    int cum = 0, ans = 0;
    for (int i = 0; i < nums.length; i++) {
      cum += nums[0];
      if (cum >= k) {
        if (map.containsKey(cum - k)) {
          ans++;
        }
      }
      map.put(cum, true);
    }
    return ans;
  }

  public boolean checkSubarraySum(int[] nums, int k) {
    if (nums.length <= 1) return false;
    for (int i = 0; i < nums.length - 1; i++) {
      if (nums[i] == 0 && nums[i + 1] == 0) return true;
    }
    if (k == 0) {
      return false;
    }
    k = Math.abs(k);
    HashMap<Integer, Integer> map = new HashMap<>();
    map.put(0, 0);
    int cum = 0;
    for (int i = 0; i < nums.length; i++) {
      cum += nums[i];
      int t = k;
      while (t <= cum) {
        int left = cum - t;
        if (map.containsKey(left)) {
          if (left == 0 && i > 0) {
            return true;
          }
          if ((i - map.get(left) > 1)) {
            return true;
          }
        }
        t += k;
      }
      map.put(cum, i);
    }
    return false;
  }
}
