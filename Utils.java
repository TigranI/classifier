package classifier;

import java.util.Comparator;


public class Utils {
	
	// descending order
	public static class CT2d<T extends Number> implements Comparator<Tuple2<T, Double>> {

		@Override
		public int compare(Tuple2<T, Double> t1,
				Tuple2<T, Double> t2) {
			if (t1._2() < t2._2()) return 1;
			if (t1._2() > t2._2()) return -1;
			return 0;
		}
	}
	
	// ascending order
	public static class CT2a<T extends Number> implements Comparator<Tuple2<T, Double>> {

		@Override
		public int compare(Tuple2<T, Double> t1,
				Tuple2<T, Double> t2) {
			if (t1._2() > t2._2()) return 1;
			if (t1._2() < t2._2()) return -1;
			return 0;
		}
	}
	
	public static class Tuple2<U, V> {
		U u;
		V v;

		public Tuple2(U u, V v) {
			this.u = u;
			this.v = v;
		}

		public U _1() {
			return u;
		}

		public V _2() {
			return v;
		}
	}

}
