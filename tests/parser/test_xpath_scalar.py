"""Regression tests for scalar XPath result handling.

XPath 1.0 functions can return scalars (strings, numbers, booleans) instead
of node-sets. Scrapling's ``Selector.xpath()`` previously mishandled these:

- String results (``string()``, ``normalize-space()``, ``concat()``, ``name()``,
  ``local-name()``) were iterated character-by-character or raised ``TypeError``.
- Numeric results (``count()``, ``number()``) raised ``TypeError`` because
  floats are not iterable.
- Boolean results (``boolean()``) raised ``TypeError`` because bools are not
  iterable.

String results are wrapped as single text Selectors (consistent with
``//text()`` behavior). Numeric and boolean scalars raise ``XPathScalarError``
because they cannot be represented as ``Selector`` objects.
"""

import pytest

from scrapling.parser import Selector, XPathScalarError

HTML = """
<html><body>
  <h1 id="main">Hello World</h1>
  <p class="a">First</p>
  <p class="a">Second</p>
  <span>Count me</span>
  <span>Count me too</span>
  <span>Count me three</span>
</body></html>
"""


class TestXPathScalarStringResults:
    """XPath functions that return string values."""

    def test_string_function_returns_text_selector(self):
        page = Selector(HTML)
        result = page.xpath("string(//h1)")
        assert len(result) == 1
        assert result[0].text == "Hello World"
        assert str(result[0]) == "Hello World"

    def test_normalize_space_function_returns_text_selector(self):
        page = Selector(HTML)
        result = page.xpath("normalize-space(//h1)")
        assert len(result) == 1
        assert result[0].text == "Hello World"

    def test_concat_function_returns_text_selector(self):
        page = Selector(HTML)
        result = page.xpath("concat('a','b')")
        assert len(result) == 1
        assert result[0].text == "ab"

    def test_name_function_returns_text_selector(self):
        page = Selector(HTML)
        result = page.xpath("name(//h1)")
        assert len(result) == 1
        assert result[0].text == "h1"

    def test_local_name_function_returns_text_selector(self):
        page = Selector(HTML)
        result = page.xpath("local-name(//h1)")
        assert len(result) == 1
        assert result[0].text == "h1"

    def test_string_function_no_match_returns_empty(self):
        page = Selector(HTML)
        result = page.xpath("string(//nonexistent)")
        assert len(result) == 0

    def test_string_function_empty_string_returns_empty(self):
        page = Selector("<html><body><h1></h1></body></html>")
        result = page.xpath("string(//h1)")
        assert len(result) == 0

    def test_existing_node_set_xpath_still_works(self):
        page = Selector(HTML)
        result = page.xpath("//h1")
        assert len(result) == 1
        assert result[0].tag == "h1"

    def test_existing_text_node_xpath_still_works(self):
        page = Selector(HTML)
        result = page.xpath("//h1/text()")
        assert len(result) == 1
        assert result[0].text == "Hello World"


class TestXPathScalarNumericResults:
    """XPath functions that return numeric values."""

    def test_count_function_raises_xpath_scalar_error(self):
        """``count()`` returns a float and must raise XPathScalarError."""
        page = Selector(HTML)
        with pytest.raises(XPathScalarError):
            page.xpath("count(//span)")

    def test_count_function_zero_raises_xpath_scalar_error(self):
        page = Selector(HTML)
        with pytest.raises(XPathScalarError):
            page.xpath("count(//nonexistent)")

    def test_number_function_raises_xpath_scalar_error(self):
        """``number()`` returns a float and must raise XPathScalarError."""
        page = Selector(HTML)
        with pytest.raises(XPathScalarError):
            page.xpath("number('42')")

    def test_number_function_nan_raises_xpath_scalar_error(self):
        page = Selector(HTML)
        with pytest.raises(XPathScalarError):
            page.xpath("number('abc')")


class TestXPathScalarBooleanResults:
    """XPath functions that return boolean values."""

    def test_boolean_function_true_raises_xpath_scalar_error(self):
        """``boolean()`` returns a bool and must raise XPathScalarError."""
        page = Selector(HTML)
        with pytest.raises(XPathScalarError):
            page.xpath("boolean(//span)")

    def test_boolean_function_false_raises_xpath_scalar_error(self):
        page = Selector(HTML)
        with pytest.raises(XPathScalarError):
            page.xpath("boolean(//nonexistent)")


class TestXPathScalarOnSelectorList:
    """Scalar XPath results when called on a SelectorList."""

    def test_selector_list_string_xpath(self):
        page = Selector(HTML)
        result = page.css("p").xpath("string(.)")
        assert len(result) == 2
        assert result[0].text == "First"
        assert result[1].text == "Second"

    def test_selector_list_count_xpath_raises_xpath_scalar_error(self):
        page = Selector(HTML)
        with pytest.raises(XPathScalarError):
            page.css("p").xpath("count(.)")

    def test_selector_list_boolean_xpath_raises_xpath_scalar_error(self):
        page = Selector(HTML)
        with pytest.raises(XPathScalarError):
            page.css("p").xpath("boolean(.)")
