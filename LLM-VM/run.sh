
export OPENAI_API_KEY=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
export OPENROUTER_API_KEY=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
export HUGGINGFACE_API_KEY=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
#export GOOGLE_API_KEY=AIzaSyBsEvy_G1L8YaoETfFhglsY82r4Y5ddcUE
#export GOOGLE_API_KEY=AIzaSyCGJHY26wGYIdF26cOdgxgnFnQk1AGTNs8
# export GOOGLE_API_KEY=AIzaSyBW7QLEWyVlOdkTPQ6MdFMETK-oexeVJ6k
export COHERE_API_KEY=A9gMJvmb1PbkseqVQyhZA2bsy7tPjQ6DOvNXENs6


# python experiment/experiment_infra/semantic_xpath/semantic_xpath_experiment_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/itinerary_experiment.yaml
# python experiment/experiment_infra/semantic_xpath/semantic_xpath_experiment_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/todolist_gt.yaml
# python experiment/experiment_infra/semantic_xpath/semantic_xpath_experiment_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/nutrition_experiment.yaml

# # python experiment/experiment_infra/in_context/in_context_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/todolist_gt.yaml
# # python experiment/experiment_infra/in_context/in_context_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/itinerary_experiment.yaml
# python experiment/experiment_infra/in_context/in_context_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/nutrition_experiment.yaml

# python experiment/experiment_infra/flat-rag/flat_rag_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/todolist_gt.yaml
# python experiment/experiment_infra/flat-rag/flat_rag_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/itinerary_experiment.yaml
# python experiment/experiment_infra/flat-rag/flat_rag_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/nutrition_experiment.yaml


# python experiment/experiment_infra/semantic_xpath/semantic_xpath_experiment_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/itinerary_version_queries.yaml
# python experiment/experiment_infra/semantic_xpath/semantic_xpath_experiment_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/nutrition_version_queries.yaml
# python experiment/experiment_infra/semantic_xpath/semantic_xpath_experiment_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/todolist_version_queries.yaml

# python experiment/experiment_infra/flat-rag/flat_rag_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/itinerary_version_queries.yaml
# python experiment/experiment_infra/flat-rag/flat_rag_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/nutrition_version_queries.yaml
# python experiment/experiment_infra/flat-rag/flat_rag_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/todolist_version_queries.yaml

# python experiment/experiment_infra/in_context/in_context_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/itinerary_version_queries.yaml
# python experiment/experiment_infra/in_context/in_context_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/nutrition_version_queries.yaml
# python experiment/experiment_infra/in_context/in_context_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/todolist_version_queries.yaml

 ### MULTI-TURN ###
python experiment/experiment_infra/semantic_xpath/semantic_xpath_experiment_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/todolist_multiturn.yaml
python experiment/experiment_infra/semantic_xpath/semantic_xpath_experiment_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/itinerary_multiturn.yaml
python experiment/experiment_infra/semantic_xpath/semantic_xpath_experiment_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/nutrition_multiturn.yaml

python experiment/experiment_infra/in_context/in_context_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/todolist_multiturn.yaml
python experiment/experiment_infra/in_context/in_context_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/itinerary_multiturn.yaml
python experiment/experiment_infra/in_context/in_context_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/nutrition_multiturn.yaml

python experiment/experiment_infra/flat-rag/flat_rag_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/todolist_multiturn.yaml
python experiment/experiment_infra/flat-rag/flat_rag_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/itinerary_multiturn.yaml
python experiment/experiment_infra/flat-rag/flat_rag_runner.py --config /Users/joeliang/Desktop/UofT/Research/LLM-VM/experiment/queries/nutrition_multiturn.yaml