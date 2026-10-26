import json
import unittest
from contract import image_input,oracle_layout_input


class InputContract(unittest.TestCase):
    def test_raw_and_public_parquet_shapes(self):
        q={"question_id":"q","question_ar":"طلب","question_en":"Request","gold_text":"secret","query_spec":{"hidden":True}}
        e={"element_id":"e","kind":"field","text_ar":"العدد\n12","bbox":[0,0,0.5,0.5],"label_ar":"العدد","value_ar":"12"}
        base={"form_id":"f","width":1280,"height":1600}
        raw={**base,"image_path":"images/f.png","elements":[e]}
        public={**base,"image":{"bytes":b"png","path":None},"elements_json":json.dumps([e])}
        a=oracle_layout_input(q,raw);b=oracle_layout_input(q,public)
        self.assertEqual(a["elements"],b["elements"])
        self.assertNotIn("gold_text",a);self.assertNotIn("query_spec",b)
        self.assertEqual(set(b["elements"][0]),{"element_id","kind","text_ar","bbox"})
        self.assertEqual(image_input(q,public)["image"],public["image"])
        self.assertIsNone(image_input(q,public)["image_path"])


if __name__=="__main__":unittest.main()
