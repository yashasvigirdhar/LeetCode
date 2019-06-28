import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

class MyPoint {
  int x, y;
  List<MyPoint> connectedPoints;

  public MyPoint(int x, int y) {
    this.x = x;
    this.y = y;
    connectedPoints = new ArrayList<>();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    MyPoint point = (MyPoint) o;
    return x == point.x &&
            y == point.y;
  }

  @Override
  public int hashCode() {
    return Objects.hash(x, y);
  }
}
