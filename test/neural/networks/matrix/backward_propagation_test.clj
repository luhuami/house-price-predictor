(ns neural.networks.matrix.backward-propagation-test
  (:require [clojure.test :refer :all]
            [neural.networks.matrix.backward-propagation :as bp]
            [clojure.core.matrix :as matrix]))

(def theta1 [[1 2] [2 3]])
(def theta2 [[2 3] [3 4]])
(def theta3 [[2 4] [5 6]])
(def a1 [[3 0] [1 2]])
(def a2 [[12 13] [13 14]])
(def a3 [[13 14] [14 15]])
(def a4 [[8 9] [17 11]])

(deftest test-calc-small-delta
  (testing ""
    (let [calc-delta #'bp/calc-small-delta
          delta [[1 2 3] [0 1 1]]
          theta [[1 1] [2 0] [2 1]]]
      (is (= (calc-delta delta (list theta a1)) [[-66.0 0.0] [0.0 -2.0]])))))

(deftest test-generate-theta-activation-pairs
  (testing ""
    (is
      (= (#'bp/generate-theta-activation-pairs (list theta1 theta2 theta3) (list a1 a2 a3 a4))
         (list (list theta2 a2) (list theta3 a3))))))

(defn mock-calc-small-delta [delta theta-activation-pair]
  (matrix/add delta (first theta-activation-pair) (second theta-activation-pair)))


(deftest test-calc-big-deltas
  (testing ""
    (with-redefs-fn {#'bp/calc-small-delta mock-calc-small-delta}
      #(is (= (#'bp/calc-small-deltas (list theta1 theta2 theta3) (list a1 a2 a3 a4) [[1 2] [3 4]])
              '([[7 7] [14 7]] [[22 25] [33 28]] [[36 41] [49 46]]))))))

(deftest test-remove-bias-for-deltas
  (testing ""
    (is (= (#'bp/remove-bias-for-deltas (list a1 a2 a3))
           (list a1 [[13] [14]] [[14] [15]])))))

(def t1 [[1 1 0 2] [2 0 1 1]])
(def t2 [[3 2 1]])
(def ac1 [[1.5 1.5 3 0] [2 1 1 0] [1 3 0 1]])
(def ac2 [[1 1 2] [2 3 1] [1 1 3]])
(def ac3 [[3] [1] [4]])
(def Y [[2] [3] [1]])

(deftest test-calc-big-deltas
  (testing ""
    (is (= (#'bp/calc-big-deltas (list t1 t2) (list ac1 ac2 ac3) Y)
           (list [[48.0 24.0 24.0 0.0] [-21.0 -57.0 -6.0 -18.0]] [[0.0 -2.0 9.0]])))))

(deftest test-calc-theta-directives
  (testing ""
    (is (= (bp/calc-theta-directives (list t1 t2) (list ac1 ac2 ac3) Y 1.5))
        (list [[16.0 8.5 8.0 1.0] [-7.0 -19.0 -1.5 -5.5]] [[1.5 1.0 3.5]]))))