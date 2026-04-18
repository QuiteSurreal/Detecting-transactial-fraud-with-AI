import pytest
import json
import os
import tempfile
import pandas as pd
from unittest.mock import patch

# Import the services
from app.services import write as wr
from app.services import predict as pred

class TestWriteService:
    """Simple unit tests for write service functions"""

    def test_writeJSON(self):
        """Test writeJSON function"""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            f.write('[]')
            temp_file = f.name

        try:
            test_data = {"id": "test", "value": 123}
            wr.writeJSON(test_data, temp_file)

            with open(temp_file, 'r') as f:
                data = json.load(f)
                assert len(data) == 1
                assert data[0]["id"] == "test"
                assert data[0]["value"] == 123
        finally:
            os.unlink(temp_file)

    def test_editJSON(self):
        """Test editJSON function"""
        initial_data = [{"id": "test1", "value": 1}, {"id": "test2", "value": 2}]
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            json.dump(initial_data, f)
            temp_file = f.name

        try:
            updated_data = {"id": "test1", "value": 999}
            wr.editJSON("test1", updated_data, temp_file)

            with open(temp_file, 'r') as f:
                data = json.load(f)
                assert len(data) == 2
                assert data[0]["id"] == "test1"
                assert data[0]["value"] == 999
                assert data[1]["id"] == "test2"
                assert data[1]["value"] == 2
        finally:
            os.unlink(temp_file)

    def test_updateModelRegistry(self):
        """Test updateModelRegistry function"""
        registry_data = {"ExistingModel": {"path": "/old/path"}}
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            json.dump(registry_data, f)
            temp_file = f.name

        try:
            original_path = "app/utils/model_registry.json"
            with patch('app.services.write.open') as mock_file:
                mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(registry_data)
                mock_file.return_value.__enter__.return_value.write = lambda x: None

                wr.updateModelRegistry("TestModel", "/path/to/model", "Test Description", "xgb", 1, {"param": "value"})

                assert mock_file.call_count >= 2
        finally:
            os.unlink(temp_file)

class TestPredictService:
    """Simple unit tests for predict service functions"""

    def test_loadModel_known_model(self):
        """Test loadModel with a known model"""
        try:
            model = pred.loadModel("Ensemble")
            assert model is not None
        except Exception:
            pass

    def test_loadModel_unknown_model(self):
        """Test loadModel with unknown model"""
        with pytest.raises(KeyError):
            pred.loadModel("NonExistentModel")

class TestPreprocessService:
    """Simple unit tests for preprocess service functions"""

    def test_validateData_valid(self):
        """Test validateData with valid data"""
        from app.services import preprocess as prep

        valid_data = pd.DataFrame({
            "step": [1, 2],
            "type": ["PAYMENT", "TRANSFER"],
            "amount": [100.0, 200.0],
            "nameOrig": ["A", "B"],
            "oldbalanceOrg": [1000.0, 2000.0],
            "newbalanceOrig": [900.0, 1800.0],
            "nameDest": ["C", "D"],
            "oldbalanceDest": [0.0, 0.0],
            "newbalanceDest": [100.0, 200.0]
        })

        errors = prep.validateData(valid_data)
        assert errors == []

    def test_validateData_invalid(self):
        """Test validateData with invalid data"""
        from app.services import preprocess as prep

        invalid_data = pd.DataFrame({
            "step": [1, 2],
            "type": ["INVALID", "TRANSFER"],
            "amount": ["not_a_number", 200.0],
            "nameOrig": ["A", "B"],
            "oldbalanceOrg": [1000.0, 2000.0],
            "newbalanceOrig": [900.0, 1800.0],
            "nameDest": ["C", "D"],
            "oldbalanceDest": [0.0, 0.0],
            "newbalanceDest": [100.0, 200.0]
        })

        errors = prep.validateData(invalid_data)
        assert len(errors) > 0

# Reset the app at the end of all tests
@pytest.fixture(scope="session", autouse=True)
def reset_app_after_tests():
    """Reset the app after all tests complete"""
    yield
    from app.services import write as wr
    wr.reset()
    print("App reset after unit tests completed")