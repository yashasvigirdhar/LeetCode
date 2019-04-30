import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class IntervalQuestions {

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
