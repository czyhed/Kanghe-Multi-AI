"""NL2SQL 安全校验（纯函数，无外部依赖）"""
import pytest

from src.nl2sql.security import apply_role_filter, validate_sql


def test_select_is_allowed_and_gets_default_limit():
    ok, sql = validate_sql("SELECT * FROM departments")
    assert ok
    assert sql.endswith("LIMIT 100")


def test_existing_limit_is_kept():
    ok, sql = validate_sql("SELECT * FROM departments LIMIT 5;")
    assert ok
    assert sql == "SELECT * FROM departments LIMIT 5"


@pytest.mark.parametrize("sql", [
    "DELETE FROM patients",
    "UPDATE patients SET name='x'",
    "DROP TABLE patients",
    "INSERT INTO patients VALUES (1)",
])
def test_non_select_is_rejected(sql):
    ok, _ = validate_sql(sql)
    assert not ok


def test_write_keyword_inside_select_is_rejected():
    ok, _ = validate_sql("SELECT 1; DROP TABLE patients")
    assert not ok


def test_sensitive_patient_columns_are_rejected():
    ok, _ = validate_sql("SELECT phone FROM patients")
    assert not ok
    ok, _ = validate_sql("SELECT id_card FROM patients")
    assert not ok


def test_doctor_filter_injected_before_limit():
    sql = apply_role_filter("SELECT * FROM visits LIMIT 100", "doctor", department_id=3)
    assert "department_id = 3" in sql


def test_doctor_filter_added_to_existing_where():
    sql = apply_role_filter("SELECT * FROM visits WHERE age > 18", "doctor", department_id=3)
    assert sql.index("department_id = 3") < sql.index("age > 18")


def test_admin_is_not_filtered():
    original = "SELECT * FROM visits LIMIT 100"
    assert apply_role_filter(original, "admin", department_id=3) == original
