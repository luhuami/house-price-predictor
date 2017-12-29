(ns neural.networks.matrix.train-neural-networks_test
  (:require [clojure.test :refer :all]
            [neural.networks.matrix.train-neural-networks :as tnn]
            [clojure.core.matrix :as matrix]))

(deftest test-create-theta-matrix
  (testing ""
    (let [l (#'tnn/create-theta-matrix [100 25 10 3])]
      (is (count l) 3)
      (is (= (matrix/shape (first l)) [25 101]))
      (is (= (matrix/shape (second l)) [10 26]))
      (is (= (matrix/shape (last l)) [3 11])))))

(defn gen-random-epsilon-mock [_]
  0.5)

(deftest test-init-theta
  (testing ""
    (with-redefs-fn
      {#'tnn/gen-random-epsilon gen-random-epsilon-mock}
      #(is (= (#'tnn/init-theta [3 3 2])
              (list [[0.5 0.5 0.5 0.5] [0.5 0.5 0.5 0.5] [0.5 0.5 0.5 0.5]]
                    [[0.5 0.5 0.5 0.5] [0.5 0.5 0.5 0.5]]))))))

(deftest test-apply-theta-directive
  (testing ""
    (is (= (#'tnn/apply-theta-directive 2 [[2 3 4] [5 6 7]] [[1 0 1] [2 1 3]])
           [[0 3 2] [1 4 1]]))))