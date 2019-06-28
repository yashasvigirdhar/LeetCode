import java.util.Objects;

class Pair<U, V> {
  U first;
  V second;

  public Pair(U first, V second) {
    this.first = first;
    this.second = second;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Pair<?, ?> pair = (Pair<?, ?>) o;
    return (Objects.equals(first, pair.first) &&
            Objects.equals(second, pair.second));
  }

  @Override
  public int hashCode() {
    return Objects.hash(first, second);
  }
}
