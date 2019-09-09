import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class IntervalQuestions {

  public int[][] merge(int[][] intervals) {
    int n = intervals.length;
    List<int[]> res = new ArrayList<>();
    if (n == 0) return new int[0][0];
    Arrays.sort(intervals, new Comparator<int[]>() {
      @Override
      public int compare(int[] o1, int[] o2) {
        return o1[0] - o2[0];
      }
    });
    int curStart = intervals[0][0];
    int curEnd = intervals[0][1];
    boolean pending = true;
    for (int i = 1; i < intervals.length; i++) {
      if (!pending) {
        curStart = intervals[i][0];
        curEnd = intervals[i][1];
        pending = true;
        continue;
      }
      if (intervals[i][0] <= curEnd) {
        curEnd = Math.max(curEnd, intervals[i][1]);
      } else {
        res.add(new int[]{curStart, curEnd});
        pending = false;
      }
    }
    if (pending) {
      res.add(new int[]{curStart, curEnd});
    }
    int[][] ans = new int[res.size()][2];
    int idx = 0;
    for (int[] r : res) {
      ans[idx++] = r;
    }
    return ans;
  }

  public int findMinArrowShots(int[][] points) {
    if (points.length == 0) return 0;
    Arrays.sort(points, new Comparator<int[]>() {
      @Override
      public int compare(int[] o1, int[] o2) {
        int c = o1[1] - o2[1];
        if (c == 0) {
          return o1[0] - o2[0];
        }
        return c;
      }
    });
    int res = 0;
    int end = points[0][1];
    int idx = 1;
    while (idx < points.length) {
      if (points[idx][0] > end) {
        res++;
        end = points[idx][1];
      }
      idx++;
    }
    return res;
  }


  public int videoStitchingPractice(int[][] clips, int T) {
    Arrays.sort(clips, new Comparator<int[]>() {
      @Override
      public int compare(int[] o1, int[] o2) {
        return o1[0] - o2[0];
      }
    });
    int res = 0;
    int curEnd = 0;
    for (int i = 0; i < clips.length && curEnd < T; ) {
      if (clips[i][0] > curEnd) {
        return -1;
      }
      int end = curEnd;
      while (i < clips.length && clips[i][0] <= curEnd) {
        end = Math.max(end, clips[i][1]);
        i++;
      }
      curEnd = end;
      res++;
    }
    return (curEnd >= T) ? res : -1;
  }

  public List<Interval> merge(List<Interval> intervals) {
    List<Interval> ans = new ArrayList<>(intervals.size());
    if (intervals.size() == 0) {
      return ans;
    }
    intervals.sort(Comparator.comparingInt(o -> o.start));
    Interval prev = intervals.get(0);
    ans.add(prev);
    for (int i = 1; i < intervals.size(); i++) {
      Interval cur = intervals.get(i);
      if (cur.start <= prev.end) {
        prev.end = Math.max(prev.end, cur.end);
      } else {
        ans.add(cur);
        prev = cur;
      }
    }
    return ans;
  }

  public int videoStitching(int[][] clips, int T) {
    Arrays.sort(clips, Comparator.comparingInt(o -> o[0]));
    int count = 0;

    int curEnd = 0;
    for (int idx = 0; idx < clips.length && curEnd >= T; idx++) {
      if (clips[idx][0] > curEnd) {
        return -1;
      }
      int newEnd = curEnd;
      while (idx < clips.length && clips[idx][0] <= curEnd) {
        newEnd = Math.max(newEnd, clips[idx][1]);
        idx++;
      }
      count++;
      curEnd = newEnd;
    }

    return curEnd >= T ? count : -1;
  }

  public List<Interval> insert(List<Interval> intervals, Interval newI) {
    int n = intervals.size();
    List<Interval> ans = new ArrayList<>();

    int idx = 0;
    while (idx < n && newI.start > intervals.get(idx).end) {
      ans.add(intervals.get(idx));
      idx++;
    }
    if (idx == n) {
      intervals.add(newI);
      return intervals;
    }
    if (newI.end < intervals.get(idx).start) {
      // insert the new interval before idx
      ans.add(idx, newI);
      ans.addAll(intervals.subList(idx, n));
      return ans;
    } else {
      newI.start = Math.min(newI.start, intervals.get(idx).start);
    }
    int jdx = idx + 1;
    while (jdx < n && intervals.get(jdx).start <= newI.end) {
      jdx++;
    }
    newI.end = Math.max(newI.end, intervals.get(jdx - 1).end);
    ans.add(newI);

    while (jdx < n) {
      ans.add(intervals.get(jdx));
      jdx++;
    }
    return ans;

  }

  public static boolean checkIfCovered(Interval interval, List<Interval> intervals) {
    intervals.sort(Comparator.comparingInt(o -> o.end));
    int startToCover = interval.start, endToCover = interval.end;
    for (Interval i : intervals) {
      if (i.start <= startToCover) {
        if (i.end >= endToCover) {
          return true;
        } else {
          startToCover = i.end;
        }
      }
    }
    return false;
  }
}
