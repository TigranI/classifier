//package com.ayasdi.dagorlad.predict;
package classifier;


import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

public class PrimalSvmTree {

	TreeModel treemodel;

	int N;
	int Nt;
	int Nminus;
	int Nplus;
	int dim;
	double[][] X;
	int[] Y;
	double[][] K;

	int numSV;

	double deltaError;
	double splittingTh;
	double cplus;
	double cminus;
	double lambda;
	double h0;
	double h1;
	double dh;
	int k;
	double majority;

	double[][] Xt;
	double[] Yt;
	double[] Ysvm;

	NumberFormat nf;

	public PrimalSvmTree(double[][] X, int[] Y, int numSV) {

		splittingTh = 0.10;
		deltaError = 0.02;
		k = -5;
		lambda = Math.pow(10, k-2);

		this.numSV = numSV;
		this.X = X;
		this.Y = Y;
		N = X.length;
		dim = X[0].length - 1; // one dimension for b
		Nminus = 0;
		Nplus = 0;

		K = new double[N][N];

		int i = 0;
		while(i < N) {
			if(Y[i] > 0) {
				Nplus++;
			} else {
				Nminus++;
			}
			i++;
		}
		if(Nplus >= Nminus) {
			majority = 1;
		} else {
			majority = -1;
		}

		nf = NumberFormat.getInstance();
		nf.format(3);
		nf.setMaximumFractionDigits(10);

		buildModel();  // builds full model

	}

	public TreeModel getModel() {
		return treemodel;
	}

	public void buildModel() {

		treemodel = new TreeModel();

		computeK();
		List<Integer> subset = new ArrayList<Integer>();
		for(int i=0;i<N;i++) {
			subset.add(i);
		}
		int[] Ns = new int[]{N, Nplus, Nminus};
		//System.out.println("stat : " + N + "  " + Nplus + "  " + Nminus);

		treemodel.updateNs("r", Ns);
		buildTreeModel("r", subset);
		treemodel.findSVs();
	}

	public void debug() {
		//getData1();
		computeK();
		List<Integer> subset = new ArrayList<Integer>();
		for(int i=0;i<N;i++) {
			subset.add(i);
		}
		variousLosses(subset);
	}

	public void computeK() {

		for(int i=0;i<N-1;i++) {
			for(int j=i;j<N;j++) {
				K[i][j] = dot(X[i],X[j]);
				K[j][i] = K[i][j];
			}
		}
	}

	public void buildTreeModel(String address, List<Integer> subset) {

		// compute current errors, e, e+, e-
		// add them to treeModel at current address using error updateErrors function
		// decide if split is needed (both classes at current address contain at least pth% of original number of points)
		// if split is needed call optimize() which does iteration of primal() 
		// if best error is no better than error at the current address split is not made still, 
		// optimize returns either beta which is added to treeModel using updateBeta or -inf in 0th element.

		System.out.println(address + "  :  " + subset.size());

		int[] Ns = treemodel.getNs(address);
		double[] errors = computeErrors(Ns);
		treemodel.updateErrors(address, errors);
		Pair results = optimize(subset, Ns, errors);
		double[] beta = results.getBeta();
		double[] w = results.getW();

		if(w[0] > Double.NEGATIVE_INFINITY) {

			treemodel.updateBetasWsSubsets(address, beta, w, subset);
			List<Integer> subsetMinus = new ArrayList<Integer>();
			List<Integer> subsetPlus = new ArrayList<Integer>();

			for(int j : subset) {
				//double value = f(beta, subset, j);
				double value = f(w, j);
				if(value >= 0) {
					subsetPlus.add(j);
				} else {
					subsetMinus.add(j);
				}
			}

			if(subsetMinus.size() / (double) N >= splittingTh && subsetPlus.size() / (double) N >= splittingTh) {
				int[] Nvalues = new int[3];
				for(int j : subsetMinus) {
					if(Y[j] > 0) {
						Nvalues[1]++;
					} else {
						Nvalues[2]++;
					}
				}
				Nvalues[0] = subsetMinus.size();
				if(Nvalues[1] / (double) Nplus >= splittingTh && Nvalues[2] / (double) Nminus >= splittingTh) {
					String address1 = address + "-";
					treemodel.updateNs(address1, Nvalues);
					buildTreeModel(address1, subsetMinus);
				}
			}

			if(subsetMinus.size() / (double) N >= splittingTh && subsetPlus.size() / (double) N >= splittingTh) {
				int[] Nvalues = new int[3];
				for(int j : subsetPlus) {
					if(Y[j] > 0) {
						Nvalues[1]++;
					} else {
						Nvalues[2]++;
					}
				}
				Nvalues[0] = subsetPlus.size();

				if(Nvalues[1] / (double) Nplus >= splittingTh && Nvalues[2] / (double) Nminus >= splittingTh) {
					String address2 = address + "+";
					treemodel.updateNs(address2, Nvalues);
					buildTreeModel(address2, subsetPlus);
				}
			}
		}
	}

	public Pair optimize(List<Integer> subset, int[] Ns, double[] e) {

		int N = subset.size();

		// C+, C- , lambda are functions of N; lambda needs to be smaller than C+- ; lambda ~ 10^(-2) * C+-

		double[] optbeta = new double[N];
		optbeta[0] = Double.NEGATIVE_INFINITY;
		double[] optw = new double[dim+1];
		optw[0] = Double.NEGATIVE_INFINITY;
		double bestError = e[0];

		h0 = 1;
		h1 = 5;
		dh = 0.25;

		cplus = Math.pow(10, k);
		double cminus1 = Math.pow(10, k);
		double h = h0;
		while(h <= h1 && optbeta[0] == Double.NEGATIVE_INFINITY) {
			cminus = h * cminus1;
			double[] beta = primal(h, subset);
			double[] w = getW(beta, subset);
			double[] ewnew = computeWerrors(w, Ns, subset);
			if(ewnew[0] < bestError - deltaError) {
				optbeta = Arrays.copyOf(beta, N);
				optw = Arrays.copyOf(w, dim + 1);
				bestError = ewnew[0];
			}

			h += dh;
		}

		cminus = Math.pow(10, k);
		double cplus1 = Math.pow(10, k);
		h = h0;
		while(h <= h1 && optbeta[0] == Double.NEGATIVE_INFINITY) {
			cplus = h * cplus1;
			double[] beta = primal(h, subset);
			double[] w = getW(beta, subset);

			double[] ewnew = computeWerrors(w, Ns, subset);
			if(ewnew[0] < bestError - deltaError) {
				optbeta = Arrays.copyOf(beta, N);
				optw = Arrays.copyOf(w, dim + 1);
				bestError = ewnew[0];
			}		

			h += dh;
		}

		if(optw[0] == Double.NEGATIVE_INFINITY) {

			//System.out.println("Forced split ...");
			return forceSplit(subset);
		}

		return new Pair(optbeta, optw);
	}

	public Pair forceSplit(List<Integer> subset) {

		int N = subset.size();

		double[] beta = new double[N];
		beta[0] = Double.NEGATIVE_INFINITY;
		double[] w = new double[dim+1];
		w[0] = 1.0;

		Utils.CT2d<Integer> dct = new Utils.CT2d<Integer>();
		TreeSet<Utils.Tuple2<Integer, Double>> tset = new TreeSet<Utils.Tuple2<Integer, Double>>(dct);

		for(int i : subset) {
			double v = f(w, X[i]);
			Utils.Tuple2<Integer, Double> pp = new Utils.Tuple2<Integer, Double>(i, v); 
			tset.add(pp);
		}

		int count = 0;
		Iterator<Utils.Tuple2<Integer, Double>> iter = tset.iterator(); 
		while(iter.hasNext() && count < N/2) {
			iter.next();
			count++;
		}

		w[dim] = (-1) * iter.next()._2();

		return new Pair(beta, w);

	}

	public int[] computeSplits(double[] beta,  int[] Ns, List<Integer> subset) {

		int K = Ns.length;
		int[] Ns1 = new int[K];

		for(int i : subset) {
			double value = f(beta, subset, i);
			if(value > 0) {
				Ns1[1] += 1;

			} else {
				Ns1[2] += 1;
			}
		}
		Ns1[0] = Ns1[1] + Ns1[2];

		return Ns1;
	}

	public int[] computeWSplits(double[] w,  int[] Ns, List<Integer> subset) {

		int K = Ns.length;
		int[] Ns1 = new int[K];

		for(int i : subset) {
			//double value = f(beta, subset, i);
			double value = f(w, i);
			if(value > 0) {
				Ns1[1] += 1;

			} else {
				Ns1[2] += 1;
			}
		}
		Ns1[0] = Ns1[1] + Ns1[2];

		return Ns1;
	}


	public double[] computeErrors(int[] Ns) {

		int K = Ns.length;
		double[] errors = new double[K];
		int majorityClass = 1;
		if(Ns[1] >= Ns[2]) {
			majorityClass = 1;
		} else {
			majorityClass = 2;
		}
		errors[0] = Ns[3 - majorityClass]  / (double) Ns[0];
		errors[3 - majorityClass] = 1.0;
		errors[majorityClass] = 0.0;

		return errors;
	}

	public double[] computeErrors(double[] beta, int[] Ns, List<Integer> subset) {

		int K = Ns.length;
		double[] errors = new double[K];

		for(int i : subset) {
			double value = f(beta, subset, i);
			if(value * Y[i] <= 0) {
				errors[0] += 1;
				if(Y[i] > 0) {
					errors[1] += 1;
				} else {
					errors[2] += 1;
				}
			}
		}
		for(int j=0;j<K;j++) {
			errors[j] /= Ns[j];
		}

		return errors;
	}

	public double[] computeWerrors(double[] w, int[] Ns, List<Integer> subset) {

		int K = Ns.length;
		double[] errors = new double[K];

		for(int i : subset) {
			double value = dot(w, X[i]);
			if(value * Y[i] <= 0) {
				errors[0] += 1;
				if(Y[i] > 0) {
					errors[1] += 1;
				} else {
					errors[2] += 1;
				}
			}
		}
		for(int j=0;j<K;j++) {
			errors[j] /= Ns[j];
		}

		return errors;
	}

	public double[] getW(double[] beta, List<Integer> subset) {

		double[] w = new double[dim+1];

		for(int j=0;j<w.length;j++) {
			for(int ii=0;ii<subset.size();ii++) {
				int i = subset.get(ii);
				w[j] += beta[ii] * X[i][j];
			}
		}
		return w;
	}

	public double[] primal(double h, List<Integer> subset) {

		int N = subset.size();
		//	System.out.println("In primal : " + nf.format(h));

		double eps = Math.pow(10, -6);

		double[] I = new double[N];
		double[] oldGradient = new double[N];
		double[] newGradient = new double[N];
		double[] d = new double[N];
		double[] z = new double[N];
		double[] oldbeta = new double[N];		

		for(int ii=0;ii<N;ii++) {
			int i = subset.get(ii);
			d[ii] = -Y[i];
			oldGradient[ii] = -Y[i];
			if(Y[i] > 0) {
				I[ii] = cplus;
			} else {
				I[ii] = cminus;
			}
		}

		double[] beta = setInitial(subset);

		double fOld = obj(beta, subset);
		double fNew = fOld;
		double df = 0;

		int maxIterations = 25;
		int numIterations = 0;

		do {
			double t0 = minimizer1d(beta, subset, d, fNew);
			oldbeta = Arrays.copyOf(beta, beta.length);
			beta = vsum(beta, 1, d, t0);
			z = vsum(mult(beta, subset), 1, Y, -1, subset);
			I = updateI(beta, subset);
			newGradient = vsum(beta, 2 * lambda, mult3(I, z), 2);
			double m1 = mult2(newGradient, newGradient, subset);
			double m2 = mult2(oldGradient, oldGradient, subset);
			double ratio = m1 / m2;
			d = vsum(newGradient, -1, d, ratio);
			oldGradient = Arrays.copyOf(newGradient, newGradient.length);
			fNew = obj(beta, subset);
			fOld = obj(oldbeta, subset);
			df = Math.abs(fNew - fOld);
			if(fOld < fNew) {
				d = vsum(newGradient, -1, d, 0.0);  // restart
			}
			//System.out.println(numIterations + " : " + nf.format(h) + "    "  + nf.format(fNew) + "     " +  nf.format(df)  + "   " + nf.format(norm(d)) + "  --> " + 
			//	nf.format(norm(newGradient)) + "   " + nf.format(norm(beta)) + "   " + nf.format(norm(d)) + "   " + nf.format(m1) + "   " + nf.format(m2) +
			//"   " + nf.format(ratio));
			numIterations++;
		} while(df > eps && numIterations < maxIterations);

		//display("beta", beta);

		return beta;
	}

	public double[] setInitial(List<Integer> subset) {

		int K = 100;
		Random rand = new Random(0);
		int N = subset.size();
		double[] beta = new double[N];
		double[] optbeta = new double[N];
		double minloss = Double.POSITIVE_INFINITY;

		for(int j=0;j<K;j++) {
			for(int i=0;i<N;i++) {
				beta[i] = 2 * rand.nextDouble() - 1;
			}
			beta = normalize(beta);
			double loss = obj(beta, subset);
			if(loss < minloss) {
				optbeta = Arrays.copyOf(beta, beta.length);
				minloss = loss;
			}
		}

		return optbeta;
	}

	public void testError() {

		double testError = 0;
		for(int i=0;i<Yt.length;i++) {
			String address = "r";
			boolean flag = true;
			while(flag) {
				if(treemodel.betas.containsKey(address)) {

					//double[] beta = treemodel.getBeta(address);
					double[] w = treemodel.getW(address);
					//List<Integer> subset = treemodel.getSubset(address);
					//Ysvm[i] = Math.signum(f(beta, subset, Xt[i]));
					Ysvm[i] = Math.signum(f(w, Xt[i]));
					if(Ysvm[i] >= 0) {
						address += "+";
					} else {
						address += "-";
					}
					if(!treemodel.betas.containsKey(address)) {
						flag = false;
						if(Ysvm[i] * Yt[i] <= 0) {
							testError += 1;
						}
					}
				} else {
					Ysvm[i] = majority;
					flag = false;
					if(Ysvm[i] * Yt[i] <= 0) {
						testError += 1;
					}
				}
			}
		}
		testError /= Yt.length;

		System.out.println("Test error = " + nf.format(testError));
	}

	public double minimizer1d(double[] beta, List<Integer> subset, double[] d, double objf) {

		double p = 2.0;
		double dt = Math.pow(p, -1);
		double tmin = Math.pow(p, -10);
		double t0 = Math.pow(p, 10);
		double t = t0;
		double obj = 0;

		while(t >= tmin) {
			double[] w = vsum(beta, 1, d, t);
			obj = obj(w, subset);
			if(obj < objf) {
				break;
			}
			t *= dt;
		}

		return t;
	}

	public double[] updateI(double[] beta, List<Integer> subset) {

		int N = subset.size();
		double[] newI = new double[N];

		for(int ii=0;ii<N;ii++) {
			int i = subset.get(ii);
			double value = Y[i] * f(beta, subset, i);
			if(value < 1) {
				if(Y[i] < 0) {
					newI[ii] = cminus;
				} else {
					newI[ii] = cplus;
				}
			}
			else {
				newI[ii] = 0;
			}
		}

		return newI;
	}

	public void sanityCheck(double[] w0, int[] Ns, List<Integer> subset) {

		double R = 8.0;
		double[] w = new double[dim+1];
		double loss0 = objW(w0, subset);
		//double minloss = loss0;
		System.out.println("computed loss = " + nf.format(loss0));
		System.out.println("w0 = " +  nf.format(w0[0]) + "   " + nf.format(w0[1]) + "   " + nf.format(w0[2]));
		for(double i = 0;i < R;i += 1) {
			w[0] = Math.cos(2*Math.PI*i/R);
			w[1] = Math.sin(2*Math.PI*i/R);
			for(double b=-1;b<=1;b+=0.5) {
				w[2] = b;
				double loss = objW(w, subset);
				//if(loss < minloss) {
				double ratio = loss0 / loss;
				//minloss = loss;
				double[] e = computeWerrors(w, Ns, subset);
				System.out.println("w = " +  nf.format(w[0]) + "   " + nf.format(w[1]) + "   " + nf.format(w[2]) + "   error, ratio = " +  nf.format(e[0]) + "   " + nf.format(ratio));

				//}
			}
		}
	}

	public void variousLosses(List<Integer> subset) {

		double[] w = new double[dim+1];

		int k = -5;
		double h = 5;
		lambda = Math.pow(10, k-2);
		cplus = Math.pow(10, k);
		cminus = h * cplus;

		double minloss = Double.POSITIVE_INFINITY;
		double bb = 0;
		w[0] = 1;
		w[1] = 0;
		for(double b=-1;b<=1;b+=0.1) {
			w[2] = b;
			double loss = objW(w, subset);
			System.out.println(nf.format(w[2]) + "   :   " + nf.format(loss));
			if(loss < minloss) {
				minloss = loss;
				bb = b;
			}
		}

		System.out.println("best b : " + nf.format(bb));

		w[2] = bb;
		for(int i=0;i<Yt.length;i++) {
			Ysvm[i] = Math.signum(f(w, Xt[i]));
		}
	}

	//*********************
	//********************    Low level functions

	// objective for primal svm
	public double obj(double[] beta, List<Integer> subset) {

		int N = subset.size();

		double value = lambda * mult2(beta, beta, subset);

		for(int ii=0;ii<N;ii++) {
			int i = subset.get(ii);
			if(Y[i] > 0) {
				value += cplus * loss_L2(beta, subset, X[i], Y[i]);
			} else {
				value += cminus * loss_L2(beta, subset, X[i], Y[i]);
			}
		}

		return value;
	}

	// objective for primal svm for W
	public double objW(double[] w, List<Integer> subset) {

		int N = subset.size();

		double value = lambda * dot(w, w);

		for(int ii=0;ii<N;ii++) {
			int i = subset.get(ii);
			if(Y[i] > 0) {
				value += cplus * lossW_L2(w, X[i], Y[i]);
			} else {
				value += cminus * lossW_L2(w, X[i], Y[i]);
			}
		}

		return value;

	}

	public double loss_L2(double[] beta, List<Integer> subset, double[] x, double y) {

		double value = 1 - y * f(beta, subset, x);
		return Math.pow(Math.max(0, value), 2);
	}

	public double lossW_L2(double[] w, double[] x, double y) {

		double value = 1 - y * dot(w, x);
		return Math.pow(Math.max(0, value), 2);
	}

	public double lossW_Combi(double[] w, double[] x, double y) {

		double value = y * dot(w, x);
		return value;
	}

	// f(new point x)
	public double f(double[] beta, List<Integer> subset, double[] x) {

		double value = 0;
		for(int ii=0;ii<beta.length;ii++) {
			int i = subset.get(ii);
			value += beta[ii] * dot(X[i], x);
		}

		return value;
	}

	// f(train point)
	public double f(double[] beta, List<Integer> subset, int j) {

		double value = 0;
		for(int ii=0;ii<beta.length;ii++) {
			int i = subset.get(ii);
			value += beta[ii] * K[i][j];
		}

		return value;
	}

	// f(new point x)
	public double f(double[] w, double[] x) {

		double value = dot(w, x);	
		return value;
	}

	// f(train point)
	public double f(double[] w, int j) {

		double value = dot(w, X[j]);	
		return value;
	}

	// K * v, K is symmetric
	public double[] mult(double[] v, List<Integer> subset) {

		int N = v.length;
		double[] w = new double[N];

		for(int ii=0;ii<N;ii++) {
			int i = subset.get(ii);
			for(int j=0;j<N;j++) {
				w[ii] += K[i][j] * v[j];
			}
		}

		return w;
	}

	// v * K * w
	public double mult2(double[] v, double[] w, List<Integer> subset) {

		int N = subset.size();
		double value = 0;

		for(int ii=0;ii<N;ii++) {
			int i = subset.get(ii);
			for(int j=0;j<N;j++) {
				value += K[i][j] * v[ii] * w[j];
			}
		}

		return value;
	}

	// returns v * w elementwise
	public double[] mult3(double[] v, double[] w) {
		int N = v.length;
		double[] u = new double[N];
		for(int i=0;i<N;i++) {
			u[i] = v[i] * w[i];
		}
		return u;
	}

	// returns a1 * v1 + a2 * v2
	public double[] vsum(double[] v1, double a1, double[] v2, double a2) {
		double[] w = new double[v1.length];
		for(int i=0;i<w.length;i++) {
			w[i] = a1 * v1[i] + a2 * v2[i];
		}
		return w;
	}

	// returns a1 * v1 + a2 * v2, v2 is of size of original data
	public double[] vsum(double[] v1, double a1, int[] v2, double a2, List<Integer> subset) {
		int N = v1.length;
		double[] w = new double[N];
		for(int ii=0;ii<N;ii++) {
			int i = subset.get(ii);
			w[ii] = a1 * v1[ii] + a2 * v2[i];
		}
		return w;
	}

	public double dot(double[] v, double[] w) {
		double value = 0;
		for(int i=0;i<v.length;i++) {
			value += v[i] * w[i];
		}
		return value;
	}

	public double[] normalize(double[] v) {
		double[] w = new double[v.length];
		double norm = Math.sqrt(dot(v, v));
		for(int i=0;i<v.length;i++) {
			w[i] = v[i] / norm;
		}
		return w;
	}

	public double norm(double[] v) {
		return Math.sqrt(dot(v, v));
	}

	class TreeModel {
		Map<String, int[]> Ns;
		Map<String, double[]> errors;
		Map<String, double[]> betas;
		Map<String, double[]> ws;
		Map<String, List<Integer>> subsets;

		Set<Integer> sv;

		public TreeModel() {
			Ns = new HashMap<String, int[]>();
			errors = new HashMap<String, double[]>();
			betas = new HashMap<String, double[]>();
			ws = new HashMap<String, double[]>();
			subsets = new HashMap<String, List<Integer>>();

			sv = new HashSet<Integer>();
		}

		public void findSVs() {

			if(numSV > 0) {
				Utils.CT2d<Integer> dct = new Utils.CT2d<Integer>();
				TreeSet<Utils.Tuple2<Integer, Double>> tset = new TreeSet<Utils.Tuple2<Integer, Double>>(dct);

				for(String address : subsets.keySet()) {
					double[] b = betas.get(address);
					List<Integer> s = subsets.get(address);
					for(int i=0;i<b.length;i++) {
						Utils.Tuple2<Integer, Double> pp = new Utils.Tuple2<Integer, Double>(s.get(i), Math.abs(b[i])); 
						tset.add(pp);
					}
				}

				Iterator<Utils.Tuple2<Integer, Double>> iter = tset.iterator(); 
				while(iter.hasNext() && sv.size() < numSV) {
					Utils.Tuple2<Integer, Double> p = iter.next();
					int index = p._1();
					sv.add(index);
				}

				if(sv.size() < numSV) {
					for(int i=0;i<N;i++) {
						sv.add(i);
						if(sv.size() >= numSV) {
							break;
						}
					}
				}

			}
		}

		public void updateNs(String address, int[] Nvalues) {
			Ns.put(address, Nvalues);
		}

		public void updateErrors(String address, double[] e) {
			errors.put(address, e);
		}

		public void updateBetasWsSubsets(String address, double[] b, double[] w, List<Integer> subset) {

			betas.put(address, b);
			ws.put(address, w);
			subsets.put(address, subset);		
		}

		public int[] getNs(String address) {
			return Ns.get(address);
		}

		public double[] getErrors(String address) {
			return errors.get(address);
		}

		public double[] getBeta(String address) {
			return betas.get(address);
		}

		public double[] getW(String address) {
			return ws.get(address);
		}

		public Map<String, double[]> getAllW() {
			return ws;
		}

		public List<Integer> getSubset(String address) {
			return subsets.get(address);
		}

		public Set<Integer> getSVs() {
			return sv;
		}

		public void print() {

			System.out.println("Tree model:");
			for(String address : Ns.keySet()) {
				System.out.println(address + " : " );
				int[] Nvalues = Ns.get(address);
				System.out.println(Nvalues[0] + "   " + Nvalues[1] + "   " + Nvalues[2]);
				if(errors.keySet().contains(address)) {
					double[] e = errors.get(address);
					System.out.println(nf.format(e[0])); // + "   " + nf.format(e[1]) + "   " + nf.format(e[2]));
				}
			}
		}
	}

	class Pair {

		double[] beta;
		double[] w;

		public Pair(double[] beta, double[] w) {
			this.beta = beta;
			this.w = w;
		}

		public double[] getBeta() {
			return beta;
		}

		public double[] getW() {
			return w;
		}
	}

}
