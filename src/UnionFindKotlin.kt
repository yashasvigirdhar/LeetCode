class UnionFindKotlin(size: Int) {

    private val components: IntArray

    init {
        components = IntArray(size) { i -> i }
    }

    private fun findRoot(value: Int): Int {
        var root = value;
        while (components[root] != root) {
            root = components[root]
        }
        return root
    }

    fun checkIfConnected(v1: Int, v2: Int): Boolean {
        return (findRoot(v1) == findRoot(v2))
    }

    fun union(v1: Int, v2: Int) {
        val root1 = findRoot(v1)
        val root2 = findRoot(v2)
        components[root1] = root2
    }

    fun printContents() {
        print(components.asList())
    }
}