import java.util.*;
import java.util.stream.Collectors;

public class Strings {

  public List<String> topKFrequent(String[] words, int k) {

    Map<String, Integer> counts = new HashMap<>();
    for (String word : words) {
      counts.put(word, counts.getOrDefault(word, 0) + 1);
    }

    PriorityQueue<Pair<Integer, String>> pq = new PriorityQueue<>(new Comparator<Pair<Integer, String>>() {
      @Override
      public int compare(Pair<Integer, String> o1, Pair<Integer, String> o2) {
        int c = o1.first.compareTo(o2.first);
        if (c != 0) {
          return c;
        }
        return o2.second.compareTo(o1.second);
      }
    });

    for (Map.Entry<String, Integer> entry : counts.entrySet()) {
      Integer curCount = entry.getValue();
      String curString = entry.getKey();
      Pair<Integer, String> curPair = new Pair<>(curCount, curString);
      if (pq.size() < k) {
        pq.add(curPair);
      }else {
        int peekCount = pq.peek().first;
        String peekString = pq.peek().second;
        if(peekCount < curCount || (peekCount == curCount && curString.compareTo(peekString) < 0)){
          pq.poll();
          pq.add(curPair);
        }
      }
    }

    List<String> res = new ArrayList<>();
    while (!pq.isEmpty()){
      res.add(pq.poll().second);
    }
    Collections.reverse(res);
    return res;
  }

  public boolean isAlienSorted(String[] words, String order) {
    Map<Character, Integer> map = new HashMap<>();
    for (int i = 0; i < order.length(); i++) {
      map.put(order.charAt(i), i);
    }
    String[] sorted = new String[words.length];
    System.arraycopy(words, 0, sorted, 0, words.length);
    Arrays.sort(sorted, new Comparator<String>() {
      @Override
      public int compare(String o1, String o2) {
        int i = 0, j = 0;
        while (i < o1.length() && j < o2.length()) {
          if (map.get(o1.charAt(i)) < map.get(o2.charAt(j))) {
            return -1;
          } else if (map.get(o1.charAt(i)) > map.get(o2.charAt(j))) {
            return 1;
          } else {
            i++;
            j++;
          }
        }
        if (i < o1.length()) {
          return 1;
        } else if (j < o2.length()) {
          return -1;
        }
        return 0;
      }
    });
    int i = 0;
    while (i < words.length) {
      if (!sorted[i].equals(words[i])) {
        return false;
      }
      i++;
    }
    return true;
  }

  public String minWindowPracticeMan(String s, String t) {
    Map<Character, Integer> desired = new HashMap<>();
    for (int i = 0; i < t.length(); i++) {
      desired.put(t.charAt(i), desired.getOrDefault(t.charAt(i), 0) + 1);
    }
    int l = 0, r = 0;
    Map<Character, Integer> current = new HashMap<>();
    int required = 0;
    String res = "";
    while (r < s.length()) {
      current.put(s.charAt(r), current.getOrDefault(s.charAt(r), 0) + 1);
      if (current.get(s.charAt(r)).equals(desired.getOrDefault(s.charAt(r), 0))) {
        required++;
      }
      while (l <= r && required == desired.size()) {
        // update res
        if (res.equals("") || res.length() > r - l + 1) {
          res = s.substring(l, r + 1);
        }
        if (current.get(s.charAt(l)).equals(desired.getOrDefault(s.charAt(l), 0))) {
          required--;
        }
        current.put(s.charAt(l), current.get(s.charAt(l)) - 1);
        l++;
      }
      r++;
    }
    return res;
  }

  public int longestRepeatingSubstringDP(String s) {
    int n = s.length();
    int[][] dp = new int[n + 1][n + 1];
    int res = 0;
    for (int i = 1; i <= n; i++) {
      for (int j = i - 1; j >= 1; j--) {
        if (s.charAt(i - 1) == s.charAt(j - 1)) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
          res = Math.max(res, dp[i][j]);
        } else {
          dp[i][j] = 0;
        }
      }
    }
    return res;
  }

  public int longestRepeatingSubstring(String S) {
    Set<String> set = new HashSet<>();
    int n = S.length();
    int res = 0;
    int b = 0, e = n - 1;
    while (b <= e) {
      int mid = (b + e) / 2;
      if (searchSubstring(S, mid)) {
        res = mid;
        b = mid + 1;
      } else {
        e = mid - 1;
      }
    }
    return res;
  }

  private boolean searchSubstring(String S, int len) {
    Set<String> set = new HashSet<>();
    int n = S.length();
    for (int i = 0; i <= n - len; i++) {
      String curStr = S.substring(i, i + len);
      if (set.contains(curStr)) {
        return true;
      }
      set.add(curStr);
    }
    return false;
  }

  public List<List<String>> findDuplicate(String[] paths) {
    Map<String, List<String>> map = new HashMap<>();
    for (String path : paths) {
      String[] parts = path.split(" ");
      String basePath = parts[0] + "/";
      for (int i = 1; i < parts.length; i++) {
        String[] filePart = parts[i].split("\\(");
        if (!map.containsKey(filePart[1])) {
          map.put(filePart[1], new ArrayList<>());
        }
        map.get(filePart[1]).add(basePath + filePart[0]);
      }
    }
    List<List<String>> res = new ArrayList<>();
    for (Map.Entry<String, List<String>> entry : map.entrySet()) {
      if (entry.getValue().size() > 1) {
        res.add(entry.getValue());
      }
    }
    return res;
  }

  public String minWindowPractice(String s, String t) {
    String res = "";
    for (int i = 0; i < s.length(); i++) {
      int idx = i, jdx = 0;
      while (idx < s.length() && jdx < t.length()) {
        if (s.charAt(idx) == t.charAt(jdx)) {
          jdx++;
        }
        idx++;
      }
      if (jdx == t.length()) {
        int newLen = idx - i + 1;
        if (newLen < res.length() || res.equals("")) {
          res = s.substring(i, idx + 1);
        }
      }
    }
    return res;
  }

  public int strStrUsingKmp(String haystack, String needle) {
    int n = haystack.length();
    char[] pat = needle.toCharArray();
    if (pat.length == 0) return 0;
    if (n == 0) return -1;
    int[] lps = new int[pat.length];
    int len = 0;
    lps[0] = 0;
    int idx = 1;
    while (idx < pat.length) {
      if (pat[idx] == pat[len]) {
        len++;
        lps[idx] = len;
        idx++;
      } else {
        if (len != 0) {
          len = lps[len - 1];
        } else {
          lps[idx] = 0;
          idx++;
        }
      }
    }
    int i = 0, j = 0;

    while (i < n) {
      if (pat[j] == haystack.charAt(i)) {
        j++;
        i++;
      } else {
        if (j != 0) {
          j = lps[j - 1];
        } else {
          i++;
        }
      }
      if (j == pat.length) {
        return i - pat.length + 1;
      }
    }
    return -1;
  }

  public int strStr(String haystack, String needle) {
    int n = haystack.length();
    int m = needle.length();
    if (m == 0) return 0;
    for (int i = 0; i <= n - m; i++) {
      int idx = i, jdx = 0;
      while (idx < haystack.length() && jdx < m) {
        if (haystack.charAt(idx) != needle.charAt(jdx)) {
          break;
        }
        idx++;
        jdx++;
      }
      if (jdx == m) {
        return i;
      }
    }
    return -1;
  }


  public String rearrangeString(String s, int n) {
    Map<Character, Integer> map = new HashMap<>();
    for (int i = 0; i < s.length(); i++) {
      map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0) + 1);
    }
    PriorityQueue<Map.Entry<Character, Integer>> pq = new PriorityQueue<>(new Comparator<Map.Entry<Character, Integer>>() {
      @Override
      public int compare(Map.Entry<Character, Integer> o1, Map.Entry<Character, Integer> o2) {
        if (o2.getValue().equals(o1.getValue())) {
          return o1.getKey().compareTo(o2.getKey());
        }
        return o2.getValue() - o1.getValue();
      }
    });
    pq.addAll(map.entrySet());
    StringBuilder sb = new StringBuilder();
    while (!pq.isEmpty()) {
      int k = n + 1;
      List<Map.Entry<Character, Integer>> temp = new ArrayList<>();
      while (k > 0 && !pq.isEmpty()) {
        Map.Entry<Character, Integer> polled = pq.poll();
        polled.setValue(polled.getValue() - 1);
        k--;
        sb.append(polled.getKey());
        temp.add(polled);
      }
      if (sb.length() == s.length()) break;
      if (k > 0) {
        return "";
      }
      for (Map.Entry<Character, Integer> entry : temp) {
        if (entry.getValue() > 0) {
          pq.add(entry);
        }
      }
    }
    return sb.toString();
  }

  public String reorganizeString(String s) {
    Map<Character, Integer> map = new HashMap<>();
    for (int i = 0; i < s.length(); i++) {
      map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0) + 1);
    }
    PriorityQueue<Map.Entry<Character, Integer>> pq = new PriorityQueue<>(new Comparator<Map.Entry<Character, Integer>>() {
      @Override
      public int compare(Map.Entry<Character, Integer> o1, Map.Entry<Character, Integer> o2) {
        return o2.getValue() - o1.getValue();
      }
    });
    pq.addAll(map.entrySet());
    StringBuilder sb = new StringBuilder();
    while (!pq.isEmpty()) {
      int k = 2;
      List<Map.Entry<Character, Integer>> temp = new ArrayList<>();
      while (k > 0 && !pq.isEmpty()) {
        Map.Entry<Character, Integer> polled = pq.poll();
        polled.setValue(polled.getValue() - 1);
        k--;
        sb.append(polled.getKey());
        temp.add(polled);
      }
      if (sb.length() == s.length()) break;
      if (k > 0) {
        return "";
      }
      for (Map.Entry<Character, Integer> entry : temp) {
        if (entry.getValue() > 0) {
          pq.add(entry);
        }
      }
    }
    return sb.toString();
  }

  public boolean areSentencesSimilarTwo(String[] words1, String[] words2, List<List<String>> pairs) {
    Map<String, Set<String>> map = new HashMap<>();
    for (List<String> pair : pairs) {
      map.putIfAbsent(pair.get(0), new HashSet<>());
      map.get(pair.get(0)).add(pair.get(1));

      map.putIfAbsent(pair.get(1), new HashSet<>());
      map.get(pair.get(1)).add(pair.get(0));
    }
    int l1 = words1.length, l2 = words2.length;
    if (l1 != l2) {
      return false;
    }
    for (int i = 0; i < l1; i++) {
      String w1 = words1[i], w2 = words2[i];
      if (w1.equals(w2)) {
        continue;
      }
      if (pathExists(w1, w2, map)) {
        continue;
      } else {
        return false;
      }
    }
    return true;
  }

  private boolean pathExists(String w1, String w2, Map<String, Set<String>> map) {
    Set<String> visited = new HashSet<>();
    Deque<String> q = new ArrayDeque<>();
    visited.add(w1);
    q.addLast(w1);
    while (!q.isEmpty()) {
      String polled = q.pollFirst();
      if (polled.equals(w2)) {
        return true;
      }
      Set<String> neighbours = map.get(polled);
      if (neighbours == null) {
        continue;
      }
      for (String neighbour : neighbours) {
        if (!visited.contains(neighbour)) {
          visited.add(neighbour);
          q.addLast(neighbour);
        }
      }
    }
    return false;
  }

  public boolean areSentencesSimilar(String[] words1, String[] words2, List<List<String>> pairs) {
    Map<String, Set<String>> map = new HashMap<>();
    for (List<String> pair : pairs) {
      map.putIfAbsent(pair.get(0), new HashSet<>());
      map.get(pair.get(0)).add(pair.get(1));

      map.putIfAbsent(pair.get(1), new HashSet<>());
      map.get(pair.get(1)).add(pair.get(0));
    }
    int l1 = words1.length, l2 = words2.length;
    if (l1 != l2) {
      return false;
    }
    for (int i = 0; i < l1; i++) {
      String w1 = words1[i], w2 = words2[i];
      if (w1.equals(w2)) {
        continue;
      }
      if ((map.containsKey(w1) && map.get(w1).contains(w2)) ||
          (map.containsKey(w2) && map.get(w2).contains(w1))) {
        continue;
      } else {
        return false;
      }
    }
    return true;
  }

  public int calculate(String s) {
    Map<Character, Integer> prio = new HashMap<>();
    prio.put('/', 2);
    prio.put('*', 2);
    prio.put('+', 3);
    prio.put('-', 3);
    prio.put('(', 4);
    Deque<Integer> nums = new ArrayDeque<>();
    Deque<Character> ops = new ArrayDeque<>();
    int i = 0;
    while (i < s.length()) {
      if (s.charAt(i) == ' ') {
        i++;
      } else if (Character.isDigit(s.charAt(i))) {
        int b = i;
        while (i < s.length() && Character.isDigit(s.charAt(i))) {
          i++;
        }
        if (Long.parseLong(s.substring(b, i)) > Integer.MAX_VALUE) {
          ops.pollFirst();
          ops.addFirst('+');
          nums.addFirst(Integer.MIN_VALUE);
        } else {
          int num = Integer.parseInt(s.substring(b, i));
          nums.addFirst(num);
        }
      } else if (s.charAt(i) == '(') {
        ops.addFirst(s.charAt(i));
        i++;
      } else if (s.charAt(i) == ')') {
        while (ops.peekFirst() != '(') {
          evaluateNextExpression(nums, ops);
        }
        ops.pollFirst();
        i++;
      } else {
        while (!ops.isEmpty() && prio.get(ops.peekFirst()) <= prio.get(s.charAt(i))) {
          evaluateNextExpression(nums, ops);
        }
        ops.addFirst(s.charAt(i));
        i++;
      }

    }
    while (!ops.isEmpty()) {
      evaluateNextExpression(nums, ops);
    }
    return nums.pollFirst();
  }

  public int calculateII(String s) {
    Map<Character, Integer> prio = new HashMap<>();
    prio.put('/', 1);
    prio.put('*', 2);
    prio.put('+', 3);
    prio.put('-', 3);
    Deque<Integer> nums = new ArrayDeque<>();
    Deque<Character> ops = new ArrayDeque<>();
    int i = 0;
    while (i < s.length()) {
      if (s.charAt(i) == ' ') {
        i++;
      } else if (Character.isDigit(s.charAt(i))) {
        int b = i;
        while (i < s.length() && Character.isDigit(s.charAt(i))) {
          i++;
        }
        int num = Integer.parseInt(s.substring(b, i));
        nums.addFirst(num);
      } else {
        while (!ops.isEmpty() && prio.get(ops.peekFirst()) < prio.get(s.charAt(i))) {
          evaluateNextExpression(nums, ops);
        }
        ops.addFirst(s.charAt(i));
        i++;
      }

    }
    while (!ops.isEmpty()) {
      evaluateNextExpression(nums, ops);
    }
    return nums.pollFirst();
  }

  private void evaluateNextExpression(Deque<Integer> nums, Deque<Character> ops) {
    int symbol = ops.pollFirst();
    int num1 = nums.pollFirst(), num2 = nums.pollFirst();
    int res = 0;
    switch (symbol) {
      case '*':
        res = num2 * num1;
        break;
      case '/':
        res = num2 / num1;
        break;
      case '+':
        res = num2 + num1;
        break;
      case '-':
        res = num2 - num1;
        break;
    }
    nums.addFirst(res);
  }

  public int compress(char[] chars) {
    int newIdx = -1;
    int idx = 0;
    while (idx < chars.length) {
      char curChar = chars[idx];
      int start = idx;
      while (idx < chars.length && chars[idx] == curChar) {
        idx++;
      }
      int len = idx - start;
      chars[++newIdx] = curChar;
      if (len > 1) {
        // store count
        Deque<Integer> stack = new ArrayDeque<>();
        while (len > 0) {
          stack.addFirst(len % 10);
          len /= 10;
        }
        while (!stack.isEmpty()) {
          chars[++newIdx] = (char) (stack.pollFirst() + '0');
        }
      }
    }
    return newIdx + 1;
  }

  public List<String> subdomainVisits(String[] cpdomains) {
    Map<String, Integer> map = new HashMap<>();
    for (String s : cpdomains) {
      String[] s1 = s.split(" ");
      String domain = s1[1];
      int count = Integer.parseInt(s1[0]);
      map.put(domain, map.getOrDefault(domain, 0) + count);
      for (int i = 1; i < domain.length(); i++) {
        if (domain.charAt(i) == '.') {
          map.put(domain.substring(i + 1), map.getOrDefault(domain.substring(i + 1), 0) + count);
        }
      }
    }
    return map.entrySet()
        .stream()
        .map(entry -> entry.getValue() + " " + entry.getKey())
        .collect(Collectors.toList());
  }

  public String decodeString(String s) {
    Deque<Pair<Integer, Integer>> nums = new ArrayDeque<>();
    Deque<Pair<String, Integer>> strings = new ArrayDeque<>();
    StringBuilder current = new StringBuilder();
    int currentBIdx = -1;
    for (int i = 0; i < s.length(); i++) {
      if (Character.isDigit(s.charAt(i))) {
        if (current.length() > 0) {
          strings.push(new Pair<>(current.toString(), currentBIdx));
          current.setLength(0);
          currentBIdx = -1;
        }
        int b = i;
        while (Character.isDigit(s.charAt(i))) {
          i++;
        }
        int n = Integer.parseInt(s.substring(b, i));
        nums.push(new Pair<>(n, i));
      } else if (s.charAt(i) == ']') {
        Pair<Integer, Integer> popped = nums.pop();
        int multiplier = popped.first - 1;
        int idx = popped.second;
        while (!strings.isEmpty() && strings.peek().second > idx) {
          current.insert(0, strings.pop().first);
        }
        String toAppend = current.toString();
        for (int c = 0; c < multiplier; c++) {
          current.append(toAppend);
        }
      } else if (s.charAt(i) == '[') {
      } else {
        if (currentBIdx == -1) {
          currentBIdx = i;
        }
        current.append(s.charAt(i));
      }
    }
    while (!strings.isEmpty()) {
      current.insert(0, strings.pop().first);
    }
    return current.toString();
  }

  public String removeDuplicates(String s) {
    char[] chars = s.toCharArray();
    int[] pos = new int[chars.length];
    pos[0] = -1;
    for (int i = 1; i < chars.length; i++) {
      if (chars[i] == '$') {
        continue;
      }
      int prev = getPrev(chars, pos, i);
      while (i < chars.length && prev >= 0 && chars[i] == chars[prev]) {
        pos[i] = prev;
        chars[i] = '$';
        chars[prev] = '$';
        i++;
        prev = getPrev(chars, pos, i);
      }
    }
    StringBuilder b = new StringBuilder();
    for (char cc : chars) {
      if (cc != '$') {
        b.append(cc);
      }
    }
    return b.toString();
  }

  private int getPrev(char[] chars, int[] pos, int i) {
    int prev = i - 1;
    while (prev >= 0 && chars[prev] == '$') {
      prev--;
    }
    return prev;
  }

  public String longestDupSubstringRabinKarp(String s) {
    int mod = 10000007;
    int[] powers = new int[s.length()];
    powers[0] = 1;
    for (int i = 1; i < s.length(); i++) {
      powers[i] = (powers[i - 1] * 26) % mod;
    }
    int b = 0, e = s.length() - 1;
    String res = "";
    while (b <= e) {
      int mid = (b + e) / 2;
      String ret = longestDupSubstringUtil(mid, s, mod, powers);
      if (ret.equals("")) {
        e = mid - 1;
      } else {
        res = ret;
        b = mid + 1;
      }
    }
    return res;
  }

  private String longestDupSubstringUtil(int l, String s, int mod, int[] powers) {
    if (l == 0) return "";
    Map<Integer, List<Integer>> set = new HashMap<>();
    int cur = 0;
    for (int i = 0; i < l; i++) {
      cur = ((cur * 26) % mod + (s.charAt(i) - 'a')) % mod;
    }
    set.put(cur, new ArrayList<>());
    set.get(cur).add(0);
    for (int i = l; i < s.length(); i++) {
      int toRemove = (powers[l - 1] * (s.charAt(i - l) - 'a')) % mod;
      cur = (((cur - toRemove + mod) % mod) * 26) % mod;
      int toAdd = s.charAt(i) - 'a';
      cur = (cur + toAdd) % mod;
      if (set.containsKey(cur)) {
        for (int startPoint : set.get(cur))
          if (s.substring(startPoint, startPoint + l).equals(s.substring(i - l + 1, i + 1))) {
            return s.substring(i - l + 1, i + 1);
          }
      } else {
        set.put(cur, new ArrayList<>());
      }
      set.get(cur).add(i - l + 1);

    }
    return "";
  }

  public String longestDupSubstringBruteForce(String s) {
    String res = "";
    Map<String, Integer> map = new HashMap<>();
    for (int i = s.length() - 1; i >= 0; i--) {
      if (i + 1 <= res.length()) {
        continue;
      }
      StringBuilder b = new StringBuilder();
      for (int j = i; j >= 0; j--) {
        b.append(s.charAt(j));
        String key = b.toString();
        if (key.length() < res.length()) {
          continue;
        }
        map.put(key, map.getOrDefault(key, 0) + 1);
        if (map.get(key) >= 2 && key.length() > res.length()) {
          res = key;
        }
      }
    }
    return res;
  }

  public List<List<String>> groupAnagramsII(String[] strs) {
    Map<String, List<String>> map = new HashMap<>();
    for (String s : strs) {
      char[] chars = s.toCharArray();
      Arrays.sort(chars);
      String sorted = new String(chars);
      if (!map.containsKey(sorted)) {
        map.put(sorted, new ArrayList<>());
      }
      map.get(sorted).add(s);
    }
    List<List<String>> res = new ArrayList<>();
    for (Map.Entry<String, List<String>> entry : map.entrySet()) {
      res.add(entry.getValue());
    }
    return res;
  }

  public boolean isValid(String s) {
    Stack<Character> st = new Stack<>();
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c == ')' || c == '}' || c == ']') {
        if (st.empty() || st.pop() != c) {
          return false;
        }
      } else {
        if (c == '(') {
          st.push(')');
        } else if (c == '{') {
          st.push('}');
        } else if (c == '[') {
          st.push(']');
        }
      }
    }
    return st.empty();
  }

  public String addStrings(String num1, String num2) {
    StringBuilder builder = new StringBuilder();
    int i = num1.length() - 1, j = num2.length() - 1;
    int carry = 0, sum;
    while (i >= 0 && j >= 0) {
      sum = (num1.charAt(i) - '0') + (num2.charAt(j) - '0') + carry;
      carry = sum / 10;
      builder.append((sum % 10) + '0');
      i--;
      j--;
    }
    while (i >= 0) {
      sum = (num1.charAt(i) - '0') + carry;
      carry = sum / 10;
      builder.append((sum % 10) + '0');
      i--;
    }
    while (j >= 0) {
      sum = (num2.charAt(j) - '0') + carry;
      carry = sum / 10;
      builder.append((sum % 10) + '0');
      j--;
    }
    if (carry > 0) {
      builder.append('1');
    }
    builder.reverse();
    return builder.toString();
  }

  public String licenseKeyFormatting(String s, int K) {
    int curCount = 0;
    StringBuilder res = new StringBuilder();
    for (int i = s.length() - 1; i >= 0; i--) {
      char curChar = s.charAt(i);
      if (curChar != '-') {
        curCount++;
        if (Character.isLetter(curChar)) {
          res.append(Character.toUpperCase(curChar));
        } else {
          res.append(curChar);
        }
        if (curCount == K) {
          res.append("-");
          curCount = 0;
        }
      }
    }
    res.reverse();

    if (res.length() > 0 && res.charAt(0) == '-') {
      return res.substring(1);
    }
    return res.toString();
  }

  public String mostCommonWord(String paragraph, String[] banned) {
    HashSet<String> ban = new HashSet<>(Arrays.asList(banned));
    int maxCount = 0;
    String ans = "";
    Map<String, Integer> counts = new HashMap<>();
    char[] chars = paragraph.toCharArray();
    StringBuilder curBuilder = new StringBuilder();
    for (char c : chars) {
      if (Character.isLetter(c)) {
        curBuilder.append(Character.toLowerCase(c));
      } else {
        if (curBuilder.length() > 0) {
          String word = curBuilder.toString();
          if (!ban.contains(word)) {
            counts.put(word, counts.getOrDefault(word, 0) + 1);
            if (counts.get(word) > maxCount) {
              maxCount = counts.get(word);
              ans = word;
            }
          }
        }
        curBuilder = new StringBuilder();
      }
    }
    if (curBuilder.length() > 0) {
      String word = curBuilder.toString();
      if (!ban.contains(word)) {
        counts.put(word, counts.getOrDefault(word, 0) + 1);
        if (counts.get(word) > maxCount) {
          maxCount = counts.get(word);
          ans = word;
        }
      }
    }
    return ans;
  }

  public String[] reorderLogFiles(String[] logs) {
    List<String> numLogs = new ArrayList<>();
    TreeMap<String, String> strLogs = new TreeMap<>();
    for (String log : logs) {
      String[] words = log.split(" ");
      if (words[1].charAt(0) >= 'a' && words[1].charAt(0) <= 'z') {
        StringBuilder b = new StringBuilder();
        for (int i = 1; i < words.length; i++) {
          b.append(words[i]).append('#');
        }
        b.append(words[0]);
        strLogs.put(b.toString(), log);
      } else {
        numLogs.add(log);
      }
    }
    String[] ans = new String[logs.length];
    int idx = 0;
    for (Map.Entry<String, String> entry : strLogs.entrySet()) {
      ans[idx++] = entry.getValue();
    }
    for (String numLog : numLogs) {
      ans[idx++] = numLog;
    }
    return ans;
  }

  public boolean rotateString(String A, String B) {
    if (A.length() != B.length()) {
      return false;
    }
    if (A.length() == 0) {
      return true;
    }
    char c = A.charAt(0);
    List<Integer> startingIndicesInB = new ArrayList<>();
    for (int i = 0; i < B.length(); i++) {
      if (B.charAt(i) == c) {
        startingIndicesInB.add(i);
      }
    }
    for (int startIdx : startingIndicesInB) {
      int i = 0, j = startIdx;
      int curCount = 0;
      while (curCount < A.length()) {
        if (A.charAt(i) != B.charAt(j)) {
          break;
        }
        curCount++;
        i++;
        j++;
        if (j == B.length()) {
          j = 0;
        }
      }
      if (curCount == A.length()) {
        return true;
      }
    }
    return false;
  }


  public int[] diStringMatchIII(String s) {
    int totalDLeft = 0;
    for (int i = 0; i < s.length(); i++) {
      if (s.charAt(i) == 'D') {
        totalDLeft++;
      }
    }
    int[] ans = new int[s.length() + 1];
    ans[0] = totalDLeft;
    int nextI = totalDLeft + 1;
    int nextD = totalDLeft - 1;
    int idx = 0;
    while (idx < s.length()) {
      if (s.charAt(idx) == 'I') {
        ans[idx + 1] = nextI++;
      } else {
        ans[idx + 1] = nextD--;
      }
      idx++;
    }
    return ans;
  }

  public int[] diStringMatchII(String s) {
    boolean[] chosen = new boolean[s.length() + 1];
    Arrays.fill(chosen, false);
    int totalDLeft = 0;
    for (int i = 0; i < s.length(); i++)
      if (s.charAt(i) == 'D') {
        totalDLeft++;
      }
    int[] ans = new int[s.length() + 1];
    int idx = -1;
    while (idx < s.length()) {
      if (idx != -1 && s.charAt(idx) == 'D') {
        totalDLeft--;
      }
      int dCount = 0, newElement = -1;
      int i = 0;
      while (i < chosen.length) {
        if (!chosen[i]) {
          if (dCount < totalDLeft) {
            dCount++;
          } else {
            // it will always land here since there always exists a solution.
            newElement = i;
            break;
          }
        }
        i++;
      }
      chosen[newElement] = true;
      ans[idx + 1] = newElement;

      idx++;
    }
    return ans;
  }


  public int[] diStringMatch(String s) {
    boolean[] chosen = new boolean[s.length() + 1];
    Arrays.fill(chosen, false);
    for (int i = 0; i < chosen.length; i++) {
      chosen[i] = true;
      List<Integer> returnedList = diStringMatchUtil(0, chosen, s, i);
      if (returnedList != null) {
        returnedList.add(0, i);
        return returnedList.stream().mapToInt(Integer::intValue).toArray();
      }
      chosen[i] = false;
    }
    return null;
  }

  private List<Integer> diStringMatchUtil(int idx, boolean[] chosen, String s, int prev) {
    if (idx == s.length()) {
      // update global string
      return new ArrayList<>();
    }
    char curChar = s.charAt(idx);

    for (int i = 0; i < chosen.length; i++) {
      if (!chosen[i]) {
        if ((curChar == 'I' && i > prev) || (curChar == 'D' && i < prev)) {
          chosen[i] = true;
          List<Integer> returnedList = diStringMatchUtil(idx + 1, chosen, s, i);
          if (returnedList != null) {
            returnedList.add(0, i);
            return returnedList;
          }
          chosen[i] = false;
        }
      }
    }
    return null;
  }


  public int uniqueLetterStringIII(String s) {
    int idx = 0;
    int[] contri = new int[26];
    int[] lastPos = new int[26];
    int result = 0;
    while (idx < s.length()) {
      int curChar = s.charAt(idx) - 'A';
      contri[curChar] = idx + 1 - lastPos[curChar];
      int cur = 0;
      lastPos[curChar] = idx + 1;
      for (int i = 0; i < 26; i++) {
        cur += contri[i];
      }
      result += cur;
    }
    return result;


  }

  public int uniqueLetterStringII(String s) {
    int[] lastPosition = new int[26];
    int[] contribution = new int[26];
    int res = 0;

//     Basically, at each it, we count the contribution of all the characters to all the substrings ending till that point.

    for (int i = 0; i < s.length(); i++) {

      int curChar = s.charAt(i) - 'A';

//       Now, we need to update the contribution of curChar.
//       The total number of substrings ending at i are i+1. So if it was a unique character, it'd contribute to all of those
//       and it's contribution would have been i+1.
//       But if it's repeating, it means it has already contributed previously. So remove it's previous contribution.
//       We can do that as we have it's last position.
//       So these are the contributions for strings which start after this character's last occurrence and end at i.
//       A simple example will demonstrate that the number of these strings are i+1 - lastPosition[curChar]
//       For characters not appeared till now, lastPosition[curChar] would be 0.
      int totalNumOfSubstringsEndingHere = i + 1;
      contribution[curChar] = totalNumOfSubstringsEndingHere - lastPosition[curChar];

//       Note that, the contribution of all the other characters will remain same.

//       count the cur answer by summing all the contributions. This loop can be avoided by the idea in original post, but I find
//       it easy to understand with this and it only iterates over 26 values.
      int cur = 0;
      for (int j = 0; j < 26; j++) {
        cur += contribution[j];
      }

//       add the current value to final answer.
      res += cur;

//       update the last position of this char. This helps in future to count it's contribution if it appears again.
      lastPosition[curChar] = i + 1;
    }
    return res;
  }

  public int uniqueLetterString(String s) {
    int[] map = new int[26];
    int count = 0;
    int mod = 1000000007;
    for (int i = 0; i < s.length(); i++) {
      Arrays.fill(map, 0);
      int curUnique = 0;
      int j = i;
      while (j < s.length()) {
        map[s.charAt(j) - 'A']++;
        if (map[s.charAt(j) - 'A'] == 1) {
          curUnique++;
        } else if (map[s.charAt(j) - 'A'] == 2) {
          curUnique--;
        }
        count = (curUnique % mod + curUnique) % mod;
        j++;
      }
    }
    return count;
  }

  public boolean backspaceCompare(String S, String T) {
    S = processBackspace(S);
    T = processBackspace(T);
    if (S.length() != T.length()) {
      return false;
    }
    int i = 0;
    while (i < S.length()) {
      if (S.charAt(i) != T.charAt(i)) {
        return false;
      }
      i++;
    }
    return true;
  }

  private String processBackspace(String s) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      if (s.charAt(i) == '#') {
        if (sb.length() > 0) {
          sb.deleteCharAt(sb.length() - 1);
        }
      } else {
        sb.append(s.charAt(i));
      }
    }
    return sb.toString();
  }

  public String addBoldTagII(String s, String[] dict) {
    TrieHashMap trie = new TrieHashMap();
    for (String word : dict) {
      trie.insert(word);
    }
    boolean[] bold = new boolean[s.length()];
    Arrays.fill(bold, false);
    int curStart;
    int idx = 0;
    while (idx < s.length()) {
      String curStr = s.substring(idx);
      // find longest prefix of curStr which exists in dict.
      String longestPrefix = trie.findLongestPrefix(curStr);
      if (longestPrefix != null) {
        curStart = idx;
        int curEnd = longestPrefix.length() + idx - 1;
        for (int i = curStart; i <= curEnd && i < s.length(); i++) {
          bold[i] = true;
        }
      }
      idx++;
    }
    StringBuilder builder = new StringBuilder();
    int start = 0;
    while (start < s.length()) {
      if (!bold[start]) {
        builder.append(s.charAt(start));
        start++;
        continue;
      }
      int end = start;
      while (end < s.length() && bold[end]) {
        end++;
      }
      builder.append("<b>");
      builder.append(s, start, end);
      builder.append("</b>");
      start = end;
    }
    return builder.toString();
  }

  public String addBoldTag(String s, String[] dict) {
    TrieHashMap trie = new TrieHashMap();
    for (String word : dict) {
      trie.insert(word);
    }
    List<Integer> indices = new ArrayList<>();
    int curStart;
    int idx = 0;
    while (idx < s.length()) {
      String curStr = s.substring(idx);
      // find longest prefix of curStr which exists in dict.
      String longestPrefix = trie.findLongestPrefix(curStr);
      if (longestPrefix != null) {
        curStart = idx;
        int curEnd = longestPrefix.length() + idx - 1;
        for (int i = idx + 1; i <= curEnd + 1 && i < s.length(); i++) {
          curStr = s.substring(i);
          longestPrefix = trie.findLongestPrefix(curStr);
          if (longestPrefix != null) {
            curEnd = Math.max(curEnd, longestPrefix.length() + i - 1);
          }
        }
        indices.add(curStart);
        indices.add(curEnd);
        idx = curEnd + 2;
      } else {
        idx++;
      }
    }
    idx = 0;
    StringBuilder builder = new StringBuilder();
    int prevStart = 0;
    while (idx < indices.size()) {
      int start = indices.get(idx++);
      int end = indices.get(idx++);
      builder.append(s, prevStart, start);
      builder.append("<b>");
      builder.append(s, start, end + 1);
      builder.append("</b>");
      prevStart = end + 1;
    }
    builder.append(s, prevStart, s.length());
    return builder.toString();
  }

  public List<Integer> partitionLabels(String S) {
    int[] indices = new int[26];
    Arrays.fill(indices, -1);
    for (int i = 0; i < S.length(); i++) {
      indices[S.charAt(i) - 'a'] = i;
    }
    List<Integer> ans = new ArrayList<>();
    int curStart = -1, curEnd = -1;
    for (int i = 0; i < S.length(); i++) {
      if (curStart == -1) {
        curStart = i;
      }
      if (indices[S.charAt(i) - 'a'] != -1) {
        curEnd = Math.max(curEnd, indices[S.charAt(i) - 'a']);
      }
      if (curEnd == S.length() - 1) {
        ans.add(curEnd - curStart + 1);
        break;
      }
      if (curEnd == i) {
        ans.add(curEnd - curStart + 1);
        curStart = -1;
        curEnd = -1;
      }
    }
    return ans;
  }

  public boolean isStrobogrammatic(String num) {
    StringBuilder builder = new StringBuilder(num.length());
    for (int i = num.length() - 1; i >= 0; i--) {
      switch (num.charAt(i)) {
        case '0':
          builder.append('0');
          break;
        case '1':
          builder.append('1');
          break;
        case '6':
          builder.append('9');
          break;
        case '8':
          builder.append('8');
          break;
        case '9':
          builder.append('6');
          break;
        default:
          return false;
      }
    }
    return num.equals(builder.toString());
  }

  public int expressiveWords(String S, String[] words) {
    int ans = 0;
    for (String word : words) {
      if (isStretchy(S, word)) {
        ans++;
      }
    }
    return ans;
  }

  private boolean isStretchy(String query, String word) {
    int i = 0, j = 0;
    while (i < query.length() && j < word.length()) {
      char curChar = query.charAt(i);
      int count1 = 0;
      while (i < query.length() && query.charAt(i) == curChar) {
        count1++;
        i++;
      }
      int count2 = 0;
      while (j < word.length() && word.charAt(j) == curChar) {
        count2++;
        j++;
      }
      if (count2 == 0) {
        return false;
      }
      if (count1 == count2) {
        continue;
      } else if (count1 < count2) {
        return false;
      } else {
        if (count1 < 3) {
          return false;
        }
      }
    }
    return i == query.length() && j == word.length();
  }

  public boolean judgeCircle(String moves) {
    int x = 0, y = 0;
    for (int i = 0; i < moves.length(); i++) {
      switch (moves.charAt(i)) {
        case 'R':
          x++;
          break;
        case 'L':
          x--;
          break;
        case 'U':
          y++;
          break;
        case 'D':
          y--;
          break;
      }
    }
    return (x == 0 && y == 0);
  }

  public int numUniqueEmails(String[] emails) {
    HashSet<String> set = new HashSet<>();
    for (String email : emails) {
      StringBuilder builder = new StringBuilder();
      int idx = 0;
      boolean domainStarted = false;
      boolean ignoreBeforeDomain = false;
      while (idx < email.length()) {
        char curChar = email.charAt(idx);
        if (curChar == '@') {
          domainStarted = true;
          builder.append(curChar);
          idx++;
          continue;
        }
        if (domainStarted) {
          builder.append(curChar);
          idx++;
          continue;
        }
        if (curChar == '.') {
          idx++;
          continue;
        }
        if (curChar == '+') {
          ignoreBeforeDomain = true;
          idx++;
          continue;
        }
        if (!ignoreBeforeDomain) {
          builder.append(curChar);
        }
        idx++;
      }
      set.add(builder.toString());
    }
    return set.size();
  }


  public List<Boolean> camelMatch(String[] queries, String pattern) {
    List<Boolean> result = new ArrayList<>();
    for (String s : queries) {
      result.add(camelMatch(s, pattern));
    }
    return result;
  }

  private boolean camelMatch(String query, String pattern) {
    int i = 0, j = 0;
    while (i < query.length() && j < pattern.length()) {
      if (query.charAt(i) == pattern.charAt(j)) {
        i++;
        j++;
        continue;
      }
      if (isLowercaseAlpha(query, i)) {
        i++;
        continue;
      }
      return false;
    }
    if (j < pattern.length()) {
      return false;
    }
    while (i < query.length()) {
      if (!isLowercaseAlpha(query, i)) {
        return false;
      }
      i++;
    }
    return true;
  }

  private boolean isLowercaseAlpha(String query, int i) {
    return query.charAt(i) >= 'a' && query.charAt(i) <= 'z';
  }

  public String removeOuterParentheses(String S) {
    int ctr = -1;
    int prev = -1;
    char[] c = S.toCharArray();
    for (int i = 0; i < c.length; i++) {
      if (c[i] == '(') {
        ctr++;
        if (ctr == 1) {
          prev = i;
        }
      } else {
        ctr--;
      }
      if (ctr == 0) {
        c[prev] = '#';
        c[i] = '#';
      }
    }
    StringBuilder builder = new StringBuilder();
    for (char aC : c) {
      if (aC != '#') {
        builder.append(aC);
      }
    }
    return builder.toString();
  }

  public String replaceWords(List<String> dict, String sentence) {
    Trie trie = new Trie();
    for (String s : dict) {
      trie.insert(s);
    }
    String[] words = sentence.split(" ");
    for (int i = 0; i < words.length; i++) {
      words[i] = trie.findUniquePrefix(words[i]);
    }
    StringBuilder builder = new StringBuilder();
    for (String word : words) {
      builder.append(word).append(" ");
    }
    return builder.substring(0, builder.length() - 1);
  }

  public List<Integer> findAnagrams(String s, String p) {
    Map<Character, Integer> map = new HashMap<>();
    for (int i = 0; i < p.length(); i++) {
      map.put(p.charAt(i), map.getOrDefault(p.charAt(i), 0) + 1);
    }
    List<Integer> ans = new ArrayList<>();
    int l = 0, r = 0;
    Map<Character, Integer> map2 = new HashMap<>();
    int formed = 0;
    while (r < s.length()) {
      char curChar = s.charAt(r);
      map2.put(curChar, map2.getOrDefault(curChar, 0) + 1);
      Integer requiredCount = map.getOrDefault(curChar, 0);
      Integer curCount = map2.getOrDefault(curChar, 0);
      if (requiredCount.equals(curCount)) {
        formed++;
      }
      while (l <= r && formed == map.size()) {
        if (map.getOrDefault(s.charAt(l), 0).equals(map2.get(s.charAt(l)))) {
          formed--;
        }
        if (s.charAt(l) == curChar) {
          curCount--;
        }
        map2.put(s.charAt(l), map2.get(s.charAt(l)) - 1);
        if (map2.get(s.charAt(l)) == 0) {
          map2.remove(s.charAt(l));
        }
        l++;
      }
      if (compareMaps(map, map2) == 0) {
        ans.add(l);
      }
      r++;
    }
    return ans;
  }

  private int compareMaps(Map<Character, Integer> map, Map<Character, Integer> map2) {
    if (map2.size() > map.size()) return 1;
    for (Map.Entry<Character, Integer> entry : map.entrySet()) {
      if (map2.getOrDefault(entry.getKey(), 0) > entry.getValue()) {
        return 1;
      } else if (map2.getOrDefault(entry.getKey(), 0) < entry.getValue()) {
        return -1;
      }
    }
    return 0;
  }

  public boolean isAnagram(String s, String t) {
    int[] counter = new int[26];
    Arrays.fill(counter, 0);
    for (int i = 0; i < s.length(); i++) {
      counter[s.charAt(i) - 'a']++;
    }
    for (int i = 0; i < t.length(); i++) {
      counter[t.charAt(i) - 'a']--;
    }
    for (int i = 0; i < 26; i++) {
      if (counter[i] != 0) return false;
    }
    return true;
  }

  private String encode(String str) {
    int[] counter = new int[26];
    Arrays.fill(counter, 0);
    for (int i = 0; i < str.length(); i++) {
      counter[str.charAt(i) - 'a']++;
    }
    StringBuilder keyBuilder = new StringBuilder();
    for (int i = 0; i < 26; i++) {
      if (counter[i] > 0) {
        keyBuilder.append(counter[i]).append('a' + i);
      }
    }
    return keyBuilder.toString();
  }

  public List<List<String>> groupAnagrams(String[] strs) {
    HashMap<String, List<String>> map = new HashMap<>();
    for (String str : strs) {
      int[] counter = new int[26];
      Arrays.fill(counter, 0);
      for (int i = 0; i < str.length(); i++) {
        counter[str.charAt(i) - 'a']++;
      }
      StringBuilder keyBuilder = new StringBuilder();
      for (int i = 0; i < 26; i++) {
        if (counter[i] > 0) {
          keyBuilder.append(counter[i]).append('a' + i);
        }
      }
      String key = keyBuilder.toString();
      if (!map.containsKey(key)) {
        map.put(key, new ArrayList<>());
      }
      map.get(key).add(str);
    }
    return new ArrayList<>(map.values());
  }

  public boolean isOneEditDistance(String s, String t) {
    for (int i = 0; i < Math.min(s.length(), t.length()); i++) {
      if (s.charAt(i) != t.charAt(i)) {
        if (s.length() == t.length()) {
          return s.substring(i + 1).equals(t.substring(i + 1));
        } else if (s.length() < t.length()) {
          return s.substring(i).equals(t.substring(i + 1));
        } else {
          return s.substring(i + 1).equals(t.substring(i));
        }
      }
    }
    return Math.abs(s.length() - t.length()) == 1;
  }

  public String findLongestWord(String s, List<String> d) {
    List<String> candidates = new ArrayList<>();
    int maxCount = 0;
    for (String str : d) {
      if (canAchieve(s, str)) {
        if (str.length() >= maxCount) {
          maxCount = str.length();
          candidates.add(str);
        }
      }
    }
    candidates.sort(String::compareTo);
    for (String str : candidates) {
      if (str.length() == maxCount) {
        return str;
      }
    }
    return "";
  }

  private boolean canAchieve(String s1, String s2) {
    int i = 0, j = 0;
    while (i < s1.length()) {
      if (s1.charAt(i) == s2.charAt(j)) {
        j++;
      }
      if (j == s2.length()) {
        return true;
      }
      i++;
    }
    return false;
  }

  public String reverseWords(String s) {
    char[] chars = s.toCharArray();
    int n = chars.length;
    int lastStart = -1;
    int i = 0;
    while (i < n) {
      if (chars[i] == ' ') {
        if (lastStart == -1) {
          continue;
        } else {
          reverseString(chars, lastStart, i - 1);
          lastStart = -1;
        }
      } else {
        if (lastStart == -1) {
          lastStart = i;
        }
      }
      i++;
    }
    if (lastStart != -1) {
      reverseString(chars, lastStart, n - 1);
    }
    return new String(chars);
  }

  public String reverseStr2(String s, int k) {
    char[] chars = s.toCharArray();
    int n = chars.length;
    if (n == 0) {
      return s;
    }
    int i = 0;
    while (i < n) {
      int j = Math.min(i + k - 1, n - 1);
      reverseString(chars, i, j);
      i += 2 * k;
    }
    return new String(chars);
  }

  public String reverseStr(String s, int k) {
    char[] chars = s.toCharArray();
    int n = chars.length;
    if (n == 0) {
      return s;
    }
    int i = 0;
    int curCount = 1, start = 0;
    while (i < n) {
      if (curCount == 2 * k) {
        reverseString(chars, start, start + k - 1);
        start = i + 1;
        curCount = 0;
      }
      i++;
      curCount++;
    }
    if (curCount > k) {
      reverseString(chars, start, start + k - 1);
    } else if (curCount > 0) {
      reverseString(chars, start, start + curCount - 2);
    }
    return new String(chars);
  }

  public void reverseString(char[] s, int b, int e) {
    int n = e - b + 1;
    if (n == 0 || n == 1) return;
    int l = b, r = e;
    while (l <= r) {
      char temp = s[l];
      s[l] = s[r];
      s[r] = temp;
      l++;
      r--;
    }
  }

  public String reverseVowels(String s) {
    if (s == null) {
      return s;
    }
    char[] chars = s.toCharArray();
    int n = chars.length;
    if (n == 0 || n == 1) {
      return s;
    }
    int l = 0, r = n - 1;
    while (l <= r) {

      while (l < r && !isVowel(chars[l])) {
        l++;
      }
      while (l < r && !isVowel(chars[r])) {
        r--;
      }
      char temp = chars[l];
      chars[l] = chars[r];
      chars[r] = temp;
      l++;
      r--;
    }
    return new String(chars);
  }

  private boolean isVowel(char charAt) {
    return charAt == 'a' || charAt == 'e' || charAt == 'i' || charAt == 'o' || charAt == 'u'
        || charAt == 'A' || charAt == 'E' || charAt == 'I' || charAt == 'O' || charAt == 'U';
  }

  public String minWindow(String s, String t) {
    int n = s.length();
    if (n == 0) {
      return "";
    }
    HashMap<Character, Integer> map = new HashMap<>();
    for (int i = 0; i < t.length(); i++) {
      Character c = t.charAt(i);
      if (!map.containsKey(c)) {
        map.put(c, 0);
      }
      map.put(c, map.get(c) + 1);
    }

    int l = 0, r = 0;
    HashMap<Character, Integer> counts = new HashMap<>();
    if (!counts.containsKey(s.charAt(0))) {
      counts.put(s.charAt(0), 0);
    }
    counts.put(s.charAt(0), counts.get(s.charAt(0)) + 1);

    int gl = -1, gr = -1, minCount = Integer.MAX_VALUE;
    boolean flag = true;
    while (flag) {

      while (r + 1 < s.length() && !isPossibleSolution(counts, map)) {
        r++;
        if (!counts.containsKey(s.charAt(r))) {
          counts.put(s.charAt(r), 0);
        }
        counts.put(s.charAt(r), counts.get(s.charAt(r)) + 1);
      }
      if (r + 1 == s.length()) {
        flag = false;
      }

      while (l <= r && isPossibleSolution(counts, map)) {
        int curCount;
        curCount = r - l + 1;
        if (curCount < minCount) {
          minCount = curCount;
          gl = l;
          gr = r;
        }
        counts.put(s.charAt(l), counts.get(s.charAt(l)) - 1);
        l++;
      }
    }
    if (isPossibleSolution(counts, map)) {
      int curCount = r - l + 1;
      if (curCount < minCount) {
        minCount = curCount;
        gl = (l == 0) ? 0 : l - 1;
        gr = r;
      }
    }
    return (minCount == Integer.MAX_VALUE) ? "" : s.substring(gl, gr + 1);
  }

  private boolean isPossibleSolution(HashMap<Character, Integer> count, HashMap<Character, Integer> map) {
    boolean possibleSolution = true;
    for (Map.Entry<Character, Integer> entry : map.entrySet()) {
      if (!count.containsKey(entry.getKey()) || count.get(entry.getKey()) < entry.getValue()) {
        possibleSolution = false;
        break;
      }
    }
    return possibleSolution;
  }
}
