import java.util.*;

public class TwoPointer {

  public int subarraysWithKDistinct2(int[] A, int K) {
    return subarraysWithKOrLessDistinct(A, K) - subarraysWithKOrLessDistinct(A, K - 1);
  }

  public int subarraysWithKOrLessDistinct(int[] A, int K) {
    int n = A.length;
    if (n == 0) return 0;
    int l = 0, r = 0, ans = 0;
    Map<Integer, Integer> map = new HashMap<>();
    while (r < A.length) {
      map.put(A[r], map.getOrDefault(A[r], 0) + 1);
      while (l <= r && map.size() > K) {
        map.put(A[l], map.get(A[l]) - 1);
        if (map.get(A[l]) == 0) {
          map.remove(A[l]);
        }
        l++;
      }
      ans += (r=l+1);
      r++;
    }
    return ans;
  }

  public int characterReplacement2(String s, int k) {
    int n = s.length();
    if (n == 0) return 0;
    int l = 0, r = 0;
    HashMap<Character, Integer> map = new HashMap<>();
    int maxCount = 0;
    while (r < s.length()) {
      map.put(s.charAt(r), map.getOrDefault(s.charAt(r), 0) + 1);
      while (l <= r && ((r - l + 1) - maxElementValueInMap(map) > k)) {
        map.put(s.charAt(l), map.get(s.charAt(l)) - 1);
        l++;
      }
      maxCount = Math.max(maxCount, r - l + 1);
      r++;
    }
    return maxCount;
  }

  private int maxElementValueInMap(HashMap<Character, Integer> map) {
    int max = 0;
    for (Map.Entry<Character, Integer> entry : map.entrySet()) {
      max = Math.max(max, entry.getValue());
    }
    return max;
  }

  public int characterReplacement(String s, int k) {
    int n = s.length();
    if (n == 0) return 0;
    int l = 0, r = 0;
    Map<Character, Integer> map = new HashMap<>();
    int majorityCount = 0, maxCount = 0;
    while (r < s.length()) {
      map.put(s.charAt(r), map.getOrDefault(s.charAt(r), 0) + 1);
      if (map.get(s.charAt(r)) > majorityCount) {
        majorityCount = map.get(s.charAt(r));
      } else {
        k--;
      }
      while (l <= r && k < 0) {
        Integer prevValue = map.get(s.charAt(l));
        map.put(s.charAt(l), prevValue - 1);
        if (prevValue == majorityCount && !map.containsValue(majorityCount)) {
          majorityCount--;
        } else {
          k++;
        }
        l++;
      }
      maxCount = Math.max(maxCount, r - l + 1);
      r++;
    }
    return maxCount;
  }


  public int longestOnes(int[] nums, int K) {
    int n = nums.length;
    if (n == 0) return 0;
    int l = 0, r = 0;
    int numZeroes = 0, maxCount = 0;
    while (r < n) {
      if (nums[r] == 0) {
        numZeroes++;
      }
      while (l <= r && numZeroes > K) {
        if (nums[l] == 0) {
          numZeroes--;
        }
        l++;
      }
      maxCount = Math.max(maxCount, r - l + 1);
      r++;
    }
    return maxCount;
  }

  public int findMaxConsecutiveOnes(int[] nums) {
    int n = nums.length;
    if (n == 0) return 0;
    int l = 0, r = 0;
    int numZeroes = 0, maxCount = 0;
    while (r < n) {
      if (nums[r] == 0) {
        numZeroes++;
      }
      while (l <= r && numZeroes > 1) {
        if (nums[l] == 0) {
          numZeroes--;
        }
        l++;
      }
      maxCount = Math.max(maxCount, r - l + 1);
      r++;
    }
    return maxCount;
  }

  public int subarraysWithKDistinct(int[] A, int K) {
    int size = A.length;
    if (size == 0) return 0;
    int l, r;
    int ans = 0;
    Set<List<Integer>> set = new HashSet<>();
    for (int i = 0; i < A.length - 1; i++) {
      l = i;
      r = i;
      Map<Integer, Integer> map = new HashMap<>();
      while (r < A.length) {
        map.put(A[r], map.getOrDefault(A[r], 0) + 1);
        while (l <= r && map.size() > K) {
          map.put(A[l], map.get(A[l]) - 1);
          if (map.get(A[l]) == 0) {
            map.remove(A[l]);
          }
          l++;
        }
        if (map.size() == K) {
          List<Integer> curList = new ArrayList<>(2);
          curList.add(l);
          curList.add(r);
          if (!set.contains(curList)) {
            set.add(curList);
            ans++;
          }
        }
        r++;
      }
    }
    return ans;
  }

  public int lengthOfLongestSubstringKDistinct(String s, int k) {
    int n = s.length();
    if (n == 0) {
      return 0;
    }
    int l = 0, r = 0;
    int maxValue = 0;
    Map<Character, Integer> map = new HashMap<>();
    while (r < s.length()) {
      map.put(s.charAt(r), map.getOrDefault(s.charAt(r), 0) + 1);
      while (l <= r && map.size() > k) {
        map.put(s.charAt(l), map.get(s.charAt(l)) - 1);
        if (map.get(s.charAt(l)) == 0) {
          map.remove(s.charAt(l));
        }
        l++;
      }
      int curCount = r - l + 1;
      if (curCount > maxValue) {
        maxValue = curCount;
      }
      r++;
    }
    return maxValue;
  }

  public int lengthOfLongestSubstring(String s) {
    int n = s.length();
    if (n == 0) {
      return 0;
    }
    int l = 0, r = 0;
    int maxValue = 0;
    Map<Character, Integer> map = new HashMap<>();
    int repeating = 0;
    while (r < s.length()) {
      int toPut = map.getOrDefault(s.charAt(r), 0) + 1;
      map.put(s.charAt(r), toPut);
      if (toPut == 2) {
        repeating++;
      }
      while (l <= r && repeating > 0) {
        int toPut2 = map.get(s.charAt(l)) - 1;
        map.put(s.charAt(l), toPut2);
        if (toPut2 == 1) {
          repeating--;
        }
        l++;
      }
      int curCount = r - l + 1;
      if (curCount > maxValue) {
        maxValue = curCount;
      }
      r++;
    }
    return maxValue;
  }

  public int lengthOfLongestSubstringTwoDistinct(String s) {
    int n = s.length();
    if (n == 0) {
      return 0;
    }
    int l = 0, r = 0;
    int maxValue = 0;
    Map<Character, Integer> map = new HashMap<>();
    while (r < s.length()) {
      map.put(s.charAt(r), map.getOrDefault(s.charAt(r), 0) + 1);
      while (l <= r && map.size() > 2) {
        map.put(s.charAt(l), map.get(s.charAt(l)) - 1);
        if (map.get(s.charAt(l)) == 0) {
          map.remove(s.charAt(l));
        }
        l++;
      }
      int curCount = r - l + 1;
      if (curCount > maxValue) {
        maxValue = curCount;
      }
      r++;
    }
    return maxValue;
  }

  public String minWindow2(String s, String t) {
    int n = s.length();
    if (n == 0) {
      return "";
    }
    HashMap<Character, Integer> map = new HashMap<>();
    for (int i = 0; i < t.length(); i++) {
      char c = t.charAt(i);
      map.put(c, map.getOrDefault(c, 0) + 1);
    }
    int required = map.size();
    int formed = 0;
    HashMap<Character, Integer> counts = new HashMap<>();
    int l = 0, r = 0;
    int gl = -1, gr = -1, minCount = Integer.MAX_VALUE;
    while (r < s.length()) {
      char c = s.charAt(r);
      counts.put(c, counts.getOrDefault(c, 0) + 1);
      if (map.containsKey(c) && map.get(c).equals(counts.get(c))) {
        formed++;
      }

      while (l <= r && formed == required) {
        int curCount = r - l + 1;
        if (curCount < minCount) {
          minCount = curCount;
          gl = l;
          gr = r;
        }
        c = s.charAt(l);
        counts.put(c, counts.get(c) - 1);
        if (map.containsKey(c) && counts.get(c) < map.get(c)) {
          formed--;
        }
        l++;
      }
      r++;
    }
    return (minCount == Integer.MAX_VALUE) ? "" : s.substring(gl, gr + 1);
  }

  public int minSubArrayLen(int s, int[] nums) {
    if (nums.length == 0) return 0;
    int l = 0, r = 0;
    int curSum = 0;
    int minWindow = Integer.MAX_VALUE;
    while (r < nums.length) {
      curSum += nums[r];
      while (l <= r && curSum >= s) {
        minWindow = Math.min(minWindow, r - l + 1);
        curSum -= nums[l];
        l++;
      }
      r++;
    }
    return (minWindow == Integer.MAX_VALUE) ? 0 : minWindow;
  }

}
