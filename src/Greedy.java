import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class Greedy {

  public int findMinArrowShotsPractice(int[][] points) {
    int n = points.length;
    if (n == 0) {
      return 0;
    }
    int res = 1;
    Arrays.sort(points, new Comparator<int[]>() {
      @Override
      public int compare(int[] o1, int[] o2) {
        int c = o1[0] - o2[0];
        if (c == 0) {
          return o1[1] - o2[1];
        }
        return c;
      }
    });
    int curEnd = points[0][1];
    int idx = 1;
    while (idx < n) {
      if (points[idx][0] > curEnd) {
        curEnd = points[idx][1];
        res++;
      }
      idx++;
    }
    return res;
  }

  class Time {
    int val;
    boolean isStart;

    public Time(int val, boolean isStart) {
      this.val = val;
      this.isStart = isStart;
    }
  }

  public int minMeetingRooms(Interval[] intervals) {
    int n = intervals.length;
    if (n == 0) {
      return 0;
    }
    List<Time> times = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      times.add(new Time(intervals[i].start, true));
      times.add(new Time(intervals[i].end, false));
    }
    times.sort(new Comparator<Time>() {
      @Override
      public int compare(Time o1, Time o2) {
        if (o1.val != o2.val) {
          return o1.val - o2.val;
        }
        if (!o1.isStart) {
          return -1;
        }
        if (!o2.isStart) {
          return 1;
        }
        return 0;
      }
    });
    int count = 0, max = 0;
    for (int i = 0; i < times.size(); i++) {
      if (times.get(i).isStart) {
        count++;
        max = Math.max(max, count);
      } else {
        count--;
      }
    }
    return max;
  }

  public boolean canAttendMeetings(Interval[] intervals) {
    int n = intervals.length;
    if (n == 0) {
      return false;
    }
    Arrays.sort(intervals, (o1, o2) -> o1.end - o2.end);
    int end = intervals[0].end;
    int count = 1;
    for (int i = 1; i < n; i++) {
      if (intervals[i].start >= end) {
        count++;
        end = intervals[i].end;
      }
    }
    return (n - count == 0);
  }

  class Point {
    int x1, x2;

    public Point(int x1, int x2) {
      this.x1 = x1;
      this.x2 = x2;
    }
  }

  public int findMinArrowShots(int[][] points) {
    int n = points.length;
    List<Point> p = new ArrayList<>(n);
    for (int[] point : points) {
      p.add(new Point(point[0], point[1]));
    }
    p.sort(Comparator.comparingInt(o -> o.x2));
    int count = 1;
    int end = p.get(0).x2;
    for (int i = 1; i < n; i++) {
      if (p.get(i).x1 > end) {
        count++;
        end = p.get(i).x2;
      }
    }
    return count;
  }

  public int eraseOverlapIntervals3(Interval[] intervals) {
    int n = intervals.length;
    Arrays.sort(intervals, new Comparator<Interval>() {
      @Override
      public int compare(Interval o1, Interval o2) {
        return o1.end - o2.end;
      }
    });
    int end = intervals[0].end;
    int count = 1;
    for (int i = 1; i < n; i++) {
      if (intervals[i].start >= end) {
        count++;
        end = intervals[i].end;
      }
    }
    return n - count;
  }

  public int wiggleMaxLength3(int[] nums) {
    int n = nums.length;
    if (n == 0) return 0;
    int upMax = 1, downMax = 1;
    for (int i = 1; i < n; i++) {
      if (nums[i] > nums[i - 1]) {
        upMax = Math.max(upMax, downMax + 1);
      } else if (nums[i] < nums[i - 1]) {
        downMax = Math.max(downMax, upMax + 1);
      }
    }
    return Math.max(upMax, downMax);
  }

  public int wiggleMaxLength2(int[] nums) {
    int n = nums.length;
    if (n == 0) return 0;
    int[][] dp = new int[n][2];
    dp[0][0] = dp[0][1] = 1;
    for (int i = 1; i < n; i++) {
      if (nums[i] > nums[i - 1]) {
        dp[i][0] = dp[i - 1][1] + 1;
        dp[i][1] = dp[i - 1][1];
      } else if (nums[i] < nums[i - 1]) {
        dp[i][1] = dp[i - 1][0] + 1;
        dp[i][0] = dp[i - 1][0];
      } else {
        dp[i][0] = dp[i - 1][0];
        dp[i][1] = dp[i - 1][1];
      }
    }
    return Math.max(dp[n - 1][0], dp[n - 1][1]);
  }


  public int wiggleMaxLength(int[] nums) {
    return Math.max(iterate(nums, true, 1, nums[0]), iterate(nums, false, 1, nums[0]));
  }

  private int iterate(int[] nums, boolean flag, int len, int prev) {
    for (int i = 1; i < nums.length; i++) {
      if (flag) {
        if (nums[i] > prev) {
          flag = !flag;
          len++;
          prev = nums[i];
        } else if (nums[i] < prev) {
          prev = nums[i];
        }
      } else {
        if (nums[i] < prev) {
          flag = !flag;
          len++;
          prev = nums[i];
        } else if (nums[i] > prev) {
          prev = nums[i];
        }
      }
    }
    return len;
  }

  public boolean lemonadeChange(int[] bills) {
    int[] change = new int[3];
    for (int i = 0; i < change.length; i++) change[i] = 0;
    int[] values = new int[]{20, 10, 5};
    for (int bill : bills) {
      int left = bill - 5;
      for (int i = 0; i < 3; i++) {
        while (left >= values[i] && change[i] > 0) {
          left -= values[i];
          change[i]--;
        }
      }
      if (left > 0) {
        return false;
      }
      if (bill == 20) {
        change[0]++;
      } else if (bill == 10) {
        change[1]++;
      } else {
        change[2]++;
      }
    }
    return true;
  }
}
