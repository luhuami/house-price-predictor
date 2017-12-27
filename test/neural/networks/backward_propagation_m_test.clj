(ns neural.networks.backward-propagation-m-test
  (:require [clojure.test :refer :all]
            [neural.networks.backward-propagation-m :as bp]
            [clojure.core.matrix :as matrix]))

(def theta1 [[1 2] [2 3]])
(def theta2 [[2 3] [3 4]])
(def theta3 [[2 4] [5 6]])
(def a1 [[3 0] [1 2]])
(def a2 [[12 13] [13 14]])
(def a3 [[13 14] [14 15]])
(def a4 [[8 9] [17 11]])

(deftest test-calc-delta
  (testing ""
    (let [calc-delta #'bp/calc-delta
          delta [[1 2 3] [0 1 1]]
          theta [[1 1] [2 0] [2 1]]]
      (is (= (calc-delta delta (list theta a1)) [[-66.0 0.0] [0.0 -2.0]])))))

(deftest test-generate-theta-activation-pairs
  (testing ""
    (is
      (= (#'bp/generate-theta-activation-pairs (list theta1 theta2 theta3) (list a1 a2 a3 a4))
         (list (list theta2 a2) (list theta3 a3))))))

(defn mock-calc-delta [delta theta-activation-pair]
  (matrix/add delta (first theta-activation-pair) (second theta-activation-pair)))


(deftest test-calc-deltas
  (testing ""
    (with-redefs-fn {#'bp/calc-delta mock-calc-delta}
      #(is (= (#'bp/calc-deltas (list theta1 theta2 theta3) (list a1 a2 a3 a4) [[1 2] [3 4]])
              '([[7 7] [14 7]] [[22 25] [33 28]] [[36 41] [49 46]]))))))

(deftest test-remove-bias-for-deltas
  (testing ""
    (is (= (#'bp/remove-bias-for-deltas (list a1 a2 a3))
           (list a1 [[13] [14]] [[14] [15]])))))
