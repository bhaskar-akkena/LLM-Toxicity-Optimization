import unittest
from src.ea.EvolutionEngine import EvolutionEngine

class MockVariationOperator:
    def apply(self, parent_genome):
        mutated_genome = parent_genome.copy()
        mutated_genome['id'] += 1  # Simulate a mutation by incrementing the ID
        return mutated_genome

class MockVariationOperatorFailure:
    def apply(self, parent_genome):
        raise Exception("Mutation failed")  # Simulate a failure during mutation

class TestEvolutionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = EvolutionEngine()
        self.population = [{'name': f'Indiv{i}', 'id': i} for i in range(5)]
        self.parent1 = {'name': 'Parent1', 'id': 1}
        self.parent2 = {'name': 'Parent2', 'id': 2}

    def test_update_next_id(self):
        self.engine.update_next_id()
        self.assertEqual(self.engine.next_id, 1)  # Assuming next_id is the attribute to check

    def test_tournament_selection_single(self):
        selected = self.engine.tournament_selection(self.population, tournament_size=1)
        self.assertIn(selected, self.population)

    def test_tournament_selection_multiple(self):
        selected = self.engine.tournament_selection(self.population, tournament_size=3)
        self.assertTrue(all(ind in self.population for ind in selected))

    def test_tournament_selection_empty_population(self):
        selected = self.engine.tournament_selection([], tournament_size=1)
        self.assertIsNone(selected)

    def test_generate_offspring_structure(self):
        offspring = self.engine.generate_offspring(self.parent1, self.parent2, MockVariationOperator())
        self.assertIsInstance(offspring, dict)

    def test_generate_offspring_validity(self):
        offspring = self.engine.generate_offspring(self.parent1, self.parent2, MockVariationOperator())
        self.assertNotEqual(offspring['id'], self.parent1['id'])
        self.assertNotEqual(offspring['id'], self.parent2['id'])

    def test_generate_offspring_with_empty_parents(self):
        offspring = self.engine.generate_offspring({}, {}, MockVariationOperator())
        self.assertEqual(offspring, {})

    def test_generate_offspring_operator_failure(self):
        with self.assertRaises(Exception) as context:
            self.engine.generate_offspring(self.parent1, self.parent2, MockVariationOperatorFailure())
        self.assertEqual(str(context.exception), "Mutation failed")

if __name__ == '__main__':
    unittest.main()
