(ns neural.networks.backward-propagation-m-test
  (:require [clojure.test :refer :all]
            [neural.networks.backward-propagation-m :as bp]
            [clojure.core.matrix :as matrix]))

(deftest test-calc-delta
  (testing ""
    (let [calc-delta #'bp/calc-delta
          delta [[1 2 3] [0 1 1]]
          theta [[1 1] [2 0] [2 1]]
          activation [[3 0] [1 2]]]
      (is (= (calc-delta delta (list theta activation)) [[-66.0 0.0] [0.0 -2.0]])))))

(deftest test-generate-theta-activation-pairs
  (testing ""
    (let [theta1 [[1 2] [2 3]]
          theta2 [[2 3] [3 4]]
          theta3 [[2 4] [5 6]]
          a1 [[11 12] [12 13]]
          a2 [[12 13] [13 14]]
          a3 [[13 14] [14 15]]
          a4 [[8 9] [17 11]]]
      (is
        (= (#'bp/generate-theta-activation-pairs (list theta1 theta2 theta3) (list a1 a2 a3 a4))
           (list (list theta2 a2) (list theta3 a3)))))))

;(defn mock-calc-delta [delta theta-activation-pair]
;  (matrix/add delta (first theta-activation-pair) (second theta-activation-pair)))
;
;
;(deftest test-calc-deltas
;  (testing ""
;    (let)))
