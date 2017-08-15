(defproject house-price-predictor "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [dk.ative/docjure "1.11.0"]
                 [net.mikera/vectorz-clj "0.47.0"]]
  :main ^:skip-aot house-price-predictor.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
