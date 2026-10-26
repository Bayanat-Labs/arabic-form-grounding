import unittest
from score import assess,score,valid


class ScoringContract(unittest.TestCase):
    def setUp(self):
        self.gold={"question_id":"q","gold_status":"answerable","gold_text":"١٢","gold_bbox":[0.1,0.2,0.3,0.4],"gold_field_id":"field"}
    def test_digit_equivalence_and_optional_id(self):
        result=assess(self.gold,{"question_id":"q","status":"answerable","text":"12","bbox":[0.1,0.2,0.3,0.4]})
        self.assertTrue(result["joint"]);self.assertFalse(result["visible_text"])
    def test_wrong_target_is_not_hidden_by_correct_text(self):
        result=assess(self.gold,{"question_id":"q","status":"answerable","text":"12","bbox":[0.5,0.6,0.7,0.8]})
        self.assertFalse(result["joint"])
    def test_invalid_and_missing_outputs_fail(self):
        self.assertFalse(assess(self.gold,None)["joint"])
        self.assertFalse(valid({"question_id":"q","status":"answerable","text":"12","bbox":[0,0,float("nan"),1]}))
        self.assertFalse(valid({"question_id":"q","status":"missing","text":"12","bbox":None}))
        for malformed in [[],{},None,True,1]:
            self.assertFalse(valid({"question_id":"q","status":malformed,"text":None,"bbox":None}))
    def test_abstention_is_not_an_empty_answer(self):
        g={**self.gold,"gold_status":"ambiguous","gold_text":None,"gold_bbox":None,"gold_field_id":None}
        self.assertTrue(assess(g,{"question_id":"q","status":"ambiguous","text":None,"bbox":None})["joint"])
        self.assertFalse(assess(g,{"question_id":"q","status":"missing","text":None,"bbox":None})["joint"])
    def test_duplicate_and_unknown_ids_rejected(self):
        p={"question_id":"q","status":"missing","text":None,"bbox":None}
        with self.assertRaises(ValueError):score([self.gold],[p,p])
        with self.assertRaises(ValueError):score([self.gold],[{**p,"question_id":"unknown"}])


if __name__=="__main__":unittest.main()
