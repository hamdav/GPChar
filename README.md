# Thoughts during development

The `aquire_random_evaluation` and `aquire_new_evaluation` should be run in a loop with a separate thread that just aqcuires a lock when the GP and lists need to be updated. None of this async stuff, it's the wrong idea.
