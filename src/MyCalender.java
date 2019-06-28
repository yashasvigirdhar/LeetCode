import java.util.Comparator;
import java.util.Objects;
import java.util.TreeSet;

class MyCalender {

  private TreeSet<Point> points;

  public MyCalender() {
    points = new TreeSet<>(new Comparator<Point>() {
      @Override
      public int compare(Point o1, Point o2) {
        if (o1.val != o2.val) {
          return o1.val - o2.val;
        }
        if (!o1.isStart && o2.isStart) {
          return -1;
        }
        if (o1.isStart && !o2.isStart) {
          return 1;
        }
        return 0;
      }
    });
  }

  public boolean book(int start, int end) {
    int count = 0;
    for (Point p : points) {
      if (start < p.val) {
        if (count != 0) {
          return false;
        }
        if (end > p.val) {
          return false;
        }
        break;
      }
      if (p.isStart) {
        count++;
      } else {
        count--;
      }
    }
    points.add(new Point(start, true));
    points.add(new Point(end, false));
    return true;
  }

  class Point {
    int val;
    boolean isStart;

    public Point(int val, boolean isStart) {
      this.val = val;
      this.isStart = isStart;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      Point point = (Point) o;
      return val == point.val &&
              isStart == point.isStart;
    }

    @Override
    public int hashCode() {
      return Objects.hash(val, isStart);
    }
  }
}