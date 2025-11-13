#!/usr/bin/env python3
"""
Hubble Programming Language
A high-level programming language inspired by Python and Lua
Version 1.0.0
"""

import sys
import os
import re
import json
import math
import time
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import operator
import traceback
import copy
import pickle
import hashlib
import base64
import threading
import queue


# ============================================================================
# TOKEN TYPES AND LEXER
# ============================================================================

class TokenType(Enum):
    """Token types for Hubble language"""
    # Literals
    NUMBER = auto()
    STRING = auto()
    BOOLEAN = auto()
    NULL = auto()
    
    # Identifiers and Keywords
    IDENTIFIER = auto()
    
    # Keywords
    FUNC = auto()
    END = auto()
    IF = auto()
    THEN = auto()
    ELSE = auto()
    ELIF = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    DO = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    CLASS = auto()
    NEW = auto()
    THIS = auto()
    SUPER = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    TRY = auto()
    CATCH = auto()
    FINALLY = auto()
    THROW = auto()
    VAR = auto()
    CONST = auto()
    ASYNC = auto()
    AWAIT = auto()
    LAMBDA = auto()
    WITH = auto()
    MATCH = auto()
    CASE = auto()
    DEFAULT = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    POWER = auto()
    FLOOR_DIV = auto()
    
    # Comparison
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    GREATER = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    
    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Bitwise
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    BIT_NOT = auto()
    LEFT_SHIFT = auto()
    RIGHT_SHIFT = auto()
    
    # Assignment
    ASSIGN = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    MULTIPLY_ASSIGN = auto()
    DIVIDE_ASSIGN = auto()
    MODULO_ASSIGN = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    DOT = auto()
    COLON = auto()
    SEMICOLON = auto()
    ARROW = auto()
    DOUBLE_ARROW = auto()
    QUESTION = auto()
    
    # Special
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()
    
    # Range
    RANGE = auto()
    INCLUSIVE_RANGE = auto()


@dataclass
class Token:
    """Token representation"""
    type: TokenType
    value: Any
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.column})"


class Lexer:
    """Lexical analyzer for Hubble"""
    
    KEYWORDS = {
        'func': TokenType.FUNC,
        'end': TokenType.END,
        'if': TokenType.IF,
        'then': TokenType.THEN,
        'else': TokenType.ELSE,
        'elif': TokenType.ELIF,
        'while': TokenType.WHILE,
        'for': TokenType.FOR,
        'in': TokenType.IN,
        'do': TokenType.DO,
        'return': TokenType.RETURN,
        'break': TokenType.BREAK,
        'continue': TokenType.CONTINUE,
        'class': TokenType.CLASS,
        'new': TokenType.NEW,
        'this': TokenType.THIS,
        'super': TokenType.SUPER,
        'import': TokenType.IMPORT,
        'from': TokenType.FROM,
        'as': TokenType.AS,
        'try': TokenType.TRY,
        'catch': TokenType.CATCH,
        'finally': TokenType.FINALLY,
        'throw': TokenType.THROW,
        'var': TokenType.VAR,
        'const': TokenType.CONST,
        'true': TokenType.BOOLEAN,
        'false': TokenType.BOOLEAN,
        'null': TokenType.NULL,
        'and': TokenType.AND,
        'or': TokenType.OR,
        'not': TokenType.NOT,
        'async': TokenType.ASYNC,
        'await': TokenType.AWAIT,
        'lambda': TokenType.LAMBDA,
        'with': TokenType.WITH,
        'match': TokenType.MATCH,
        'case': TokenType.CASE,
        'default': TokenType.DEFAULT,
    }
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
    def error(self, message: str):
        raise SyntaxError(f"Lexer error at {self.line}:{self.column}: {message}")
    
    def peek(self, offset: int = 0) -> Optional[str]:
        """Peek at character without consuming"""
        pos = self.pos + offset
        if pos < len(self.source):
            return self.source[pos]
        return None
    
    def advance(self) -> Optional[str]:
        """Consume and return current character"""
        if self.pos < len(self.source):
            char = self.source[self.pos]
            self.pos += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            return char
        return None
    
    def skip_whitespace(self):
        """Skip whitespace except newlines"""
        while self.peek() and self.peek() in ' \t\r':
            self.advance()
    
    def skip_comment(self):
        """Skip single-line comments"""
        if self.peek() == '#':
            while self.peek() and self.peek() != '\n':
                self.advance()
    
    def read_number(self) -> Token:
        """Read number literal"""
        start_line = self.line
        start_col = self.column
        num_str = ''
        has_dot = False
        
        while self.peek() and (self.peek().isdigit() or self.peek() == '.'):
            if self.peek() == '.':
                if has_dot:
                    break
                if self.peek(1) and self.peek(1) == '.':
                    break
                has_dot = True
            num_str += self.advance()
        
        # Handle scientific notation
        if self.peek() and self.peek() in 'eE':
            num_str += self.advance()
            if self.peek() and self.peek() in '+-':
                num_str += self.advance()
            while self.peek() and self.peek().isdigit():
                num_str += self.advance()
        
        try:
            value = float(num_str) if has_dot or 'e' in num_str.lower() else int(num_str)
            return Token(TokenType.NUMBER, value, start_line, start_col)
        except ValueError:
            self.error(f"Invalid number: {num_str}")
    
    def read_string(self, quote: str) -> Token:
        """Read string literal"""
        start_line = self.line
        start_col = self.column
        self.advance()  # consume opening quote
        
        string_chars = []
        while self.peek() and self.peek() != quote:
            if self.peek() == '\\':
                self.advance()
                next_char = self.peek()
                if next_char == 'n':
                    string_chars.append('\n')
                    self.advance()
                elif next_char == 't':
                    string_chars.append('\t')
                    self.advance()
                elif next_char == 'r':
                    string_chars.append('\r')
                    self.advance()
                elif next_char == '\\':
                    string_chars.append('\\')
                    self.advance()
                elif next_char == quote:
                    string_chars.append(quote)
                    self.advance()
                elif next_char == '0':
                    string_chars.append('\0')
                    self.advance()
                else:
                    string_chars.append(self.advance())
            else:
                string_chars.append(self.advance())
        
        if not self.peek():
            self.error("Unterminated string")
        
        self.advance()  # consume closing quote
        return Token(TokenType.STRING, ''.join(string_chars), start_line, start_col)
    
    def read_identifier(self) -> Token:
        """Read identifier or keyword"""
        start_line = self.line
        start_col = self.column
        identifier = ''
        
        while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
            identifier += self.advance()
        
        token_type = self.KEYWORDS.get(identifier, TokenType.IDENTIFIER)
        
        # Handle boolean literals
        if token_type == TokenType.BOOLEAN:
            value = identifier == 'true'
            return Token(TokenType.BOOLEAN, value, start_line, start_col)
        
        return Token(token_type, identifier, start_line, start_col)
    
    def tokenize(self) -> List[Token]:
        """Tokenize the entire source"""
        while self.pos < len(self.source):
            self.skip_whitespace()
            
            if not self.peek():
                break
            
            # Comments
            if self.peek() == '#':
                self.skip_comment()
                continue
            
            # Newlines
            if self.peek() == '\n':
                token = Token(TokenType.NEWLINE, '\n', self.line, self.column)
                self.tokens.append(token)
                self.advance()
                continue
            
            start_line = self.line
            start_col = self.column
            char = self.peek()
            
            # Numbers
            if char.isdigit():
                self.tokens.append(self.read_number())
                continue
            
            # Strings
            if char in '"\'':
                self.tokens.append(self.read_string(char))
                continue
            
            # Identifiers and keywords
            if char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
                continue
            
            # Two-character operators
            two_char = char + (self.peek(1) or '')
            
            if two_char == '==':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.EQUAL, '==', start_line, start_col))
                continue
            elif two_char == '!=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.NOT_EQUAL, '!=', start_line, start_col))
                continue
            elif two_char == '<=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.LESS_EQUAL, '<=', start_line, start_col))
                continue
            elif two_char == '>=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.GREATER_EQUAL, '>=', start_line, start_col))
                continue
            elif two_char == '<<':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.LEFT_SHIFT, '<<', start_line, start_col))
                continue
            elif two_char == '>>':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.RIGHT_SHIFT, '>>', start_line, start_col))
                continue
            elif two_char == '//':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.FLOOR_DIV, '//', start_line, start_col))
                continue
            elif two_char == '**':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.POWER, '**', start_line, start_col))
                continue
            elif two_char == '..':
                self.advance()
                self.advance()
                if self.peek() == '.':
                    self.advance()
                    self.tokens.append(Token(TokenType.INCLUSIVE_RANGE, '...', start_line, start_col))
                else:
                    self.tokens.append(Token(TokenType.RANGE, '..', start_line, start_col))
                continue
            elif two_char == '=>':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.DOUBLE_ARROW, '=>', start_line, start_col))
                continue
            elif two_char == '->':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ARROW, '->', start_line, start_col))
                continue
            elif two_char == '+=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.PLUS_ASSIGN, '+=', start_line, start_col))
                continue
            elif two_char == '-=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.MINUS_ASSIGN, '-=', start_line, start_col))
                continue
            elif two_char == '*=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.MULTIPLY_ASSIGN, '*=', start_line, start_col))
                continue
            elif two_char == '/=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.DIVIDE_ASSIGN, '/=', start_line, start_col))
                continue
            elif two_char == '%=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.MODULO_ASSIGN, '%=', start_line, start_col))
                continue
            
            # Single-character operators and delimiters
            self.advance()
            
            token_map = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MULTIPLY,
                '/': TokenType.DIVIDE,
                '%': TokenType.MODULO,
                '<': TokenType.LESS,
                '>': TokenType.GREATER,
                '=': TokenType.ASSIGN,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                ',': TokenType.COMMA,
                '.': TokenType.DOT,
                ':': TokenType.COLON,
                ';': TokenType.SEMICOLON,
                '?': TokenType.QUESTION,
                '&': TokenType.BIT_AND,
                '|': TokenType.BIT_OR,
                '^': TokenType.BIT_XOR,
                '~': TokenType.BIT_NOT,
            }
            
            if char in token_map:
                self.tokens.append(Token(token_map[char], char, start_line, start_col))
            else:
                self.error(f"Unexpected character: {char}")
        
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens


# ============================================================================
# AST NODES
# ============================================================================

@dataclass
class ASTNode:
    """Base class for AST nodes"""
    line: int = 0
    column: int = 0


@dataclass
class Program(ASTNode):
    """Root program node"""
    statements: List[ASTNode] = field(default_factory=list)


@dataclass
class NumberLiteral(ASTNode):
    """Number literal"""
    value: Union[int, float] = 0


@dataclass
class StringLiteral(ASTNode):
    """String literal"""
    value: str = ""


@dataclass
class BooleanLiteral(ASTNode):
    """Boolean literal"""
    value: bool = False


@dataclass
class NullLiteral(ASTNode):
    """Null literal"""
    pass


@dataclass
class Identifier(ASTNode):
    """Variable identifier"""
    name: str = ""


@dataclass
class BinaryOp(ASTNode):
    """Binary operation"""
    left: ASTNode = None
    operator: str = ""
    right: ASTNode = None


@dataclass
class UnaryOp(ASTNode):
    """Unary operation"""
    operator: str = ""
    operand: ASTNode = None


@dataclass
class Assignment(ASTNode):
    """Variable assignment"""
    target: ASTNode = None
    value: ASTNode = None
    operator: str = "="


@dataclass
class VarDeclaration(ASTNode):
    """Variable declaration"""
    name: str = ""
    value: Optional[ASTNode] = None
    is_const: bool = False


@dataclass
class FunctionDef(ASTNode):
    """Function definition"""
    name: str = ""
    params: List[str] = field(default_factory=list)
    body: List[ASTNode] = field(default_factory=list)
    defaults: Dict[str, ASTNode] = field(default_factory=dict)
    is_async: bool = False


@dataclass
class LambdaExpr(ASTNode):
    """Lambda expression"""
    params: List[str] = field(default_factory=list)
    body: ASTNode = None


@dataclass
class FunctionCall(ASTNode):
    """Function call"""
    function: ASTNode = None
    arguments: List[ASTNode] = field(default_factory=list)
    is_await: bool = False


@dataclass
class Return(ASTNode):
    """Return statement"""
    value: Optional[ASTNode] = None


@dataclass
class IfStatement(ASTNode):
    """If statement"""
    condition: ASTNode = None
    then_branch: List[ASTNode] = field(default_factory=list)
    elif_branches: List[Tuple[ASTNode, List[ASTNode]]] = field(default_factory=list)
    else_branch: List[ASTNode] = field(default_factory=list)


@dataclass
class WhileLoop(ASTNode):
    """While loop"""
    condition: ASTNode = None
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class ForLoop(ASTNode):
    """For loop"""
    variable: str = ""
    iterable: ASTNode = None
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class Break(ASTNode):
    """Break statement"""
    pass


@dataclass
class Continue(ASTNode):
    """Continue statement"""
    pass


@dataclass
class ArrayLiteral(ASTNode):
    """Array literal"""
    elements: List[ASTNode] = field(default_factory=list)


@dataclass
class DictLiteral(ASTNode):
    """Dictionary literal"""
    pairs: List[Tuple[ASTNode, ASTNode]] = field(default_factory=list)


@dataclass
class IndexAccess(ASTNode):
    """Array/Dict index access"""
    object: ASTNode = None
    index: ASTNode = None


@dataclass
class MemberAccess(ASTNode):
    """Object member access"""
    object: ASTNode = None
    member: str = ""


@dataclass
class ClassDef(ASTNode):
    """Class definition"""
    name: str = ""
    superclass: Optional[str] = None
    methods: List[FunctionDef] = field(default_factory=list)
    properties: List[VarDeclaration] = field(default_factory=list)


@dataclass
class NewInstance(ASTNode):
    """Create new class instance"""
    class_name: str = ""
    arguments: List[ASTNode] = field(default_factory=list)


@dataclass
class This(ASTNode):
    """This reference"""
    pass


@dataclass
class Super(ASTNode):
    """Super reference"""
    pass


@dataclass
class ImportStatement(ASTNode):
    """Import statement"""
    module: str = ""
    items: List[str] = field(default_factory=list)
    alias: Optional[str] = None


@dataclass
class TryStatement(ASTNode):
    """Try-catch-finally statement"""
    try_block: List[ASTNode] = field(default_factory=list)
    catch_var: Optional[str] = None
    catch_block: List[ASTNode] = field(default_factory=list)
    finally_block: List[ASTNode] = field(default_factory=list)


@dataclass
class ThrowStatement(ASTNode):
    """Throw statement"""
    value: ASTNode = None


@dataclass
class RangeExpr(ASTNode):
    """Range expression"""
    start: ASTNode = None
    end: ASTNode = None
    inclusive: bool = False


@dataclass
class MatchStatement(ASTNode):
    """Match statement (pattern matching)"""
    value: ASTNode = None
    cases: List[Tuple[ASTNode, List[ASTNode]]] = field(default_factory=list)
    default_case: List[ASTNode] = field(default_factory=list)


@dataclass
class WithStatement(ASTNode):
    """With statement (context manager)"""
    context: ASTNode = None
    variable: Optional[str] = None
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class TernaryOp(ASTNode):
    """Ternary conditional operator"""
    condition: ASTNode = None
    true_value: ASTNode = None
    false_value: ASTNode = None


# ============================================================================
# PARSER
# ============================================================================

class Parser:
    """Parser for Hubble language"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = [t for t in tokens if t.type != TokenType.COMMENT]
        self.pos = 0
        self.current_token = self.tokens[0] if self.tokens else None
    
    def error(self, message: str):
        if self.current_token:
            raise SyntaxError(
                f"Parse error at {self.current_token.line}:{self.current_token.column}: {message}"
            )
        raise SyntaxError(f"Parse error: {message}")
    
    def peek(self, offset: int = 0) -> Optional[Token]:
        """Peek at token without consuming"""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None
    
    def advance(self) -> Token:
        """Consume and return current token"""
        token = self.current_token
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None
        return token
    
    def expect(self, token_type: TokenType) -> Token:
        """Consume token of expected type"""
        if not self.current_token or self.current_token.type != token_type:
            self.error(f"Expected {token_type}, got {self.current_token}")
        return self.advance()
    
    def skip_newlines(self):
        """Skip newline tokens"""
        while self.current_token and self.current_token.type == TokenType.NEWLINE:
            self.advance()
    
    def parse(self) -> Program:
        """Parse program"""
        statements = []
        self.skip_newlines()
        
        while self.current_token and self.current_token.type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()
        
        return Program(statements=statements)
    
    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a statement"""
        self.skip_newlines()
        
        if not self.current_token or self.current_token.type == TokenType.EOF:
            return None
        
        token_type = self.current_token.type
        
        # Function definition
        if token_type == TokenType.FUNC:
            return self.parse_function_def()
        
        # Class definition
        if token_type == TokenType.CLASS:
            return self.parse_class_def()
        
        # Variable declaration
        if token_type in (TokenType.VAR, TokenType.CONST):
            return self.parse_var_declaration()
        
        # Control flow
        if token_type == TokenType.IF:
            return self.parse_if_statement()
        
        if token_type == TokenType.WHILE:
            return self.parse_while_loop()
        
        if token_type == TokenType.FOR:
            return self.parse_for_loop()
        
        if token_type == TokenType.RETURN:
            return self.parse_return()
        
        if token_type == TokenType.BREAK:
            self.advance()
            return Break()
        
        if token_type == TokenType.CONTINUE:
            self.advance()
            return Continue()
        
        # Import
        if token_type == TokenType.IMPORT:
            return self.parse_import()
        
        # Try-catch
        if token_type == TokenType.TRY:
            return self.parse_try_statement()
        
        # Throw
        if token_type == TokenType.THROW:
            return self.parse_throw()
        
        # Match
        if token_type == TokenType.MATCH:
            return self.parse_match_statement()
        
        # With
        if token_type == TokenType.WITH:
            return self.parse_with_statement()
        
        # Expression statement
        return self.parse_expression_statement()
    
    def parse_function_def(self) -> FunctionDef:
        """Parse function definition"""
        is_async = False
        if self.current_token.type == TokenType.ASYNC:
            is_async = True
            self.advance()
        
        self.expect(TokenType.FUNC)
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value
        
        self.expect(TokenType.LPAREN)
        params = []
        defaults = {}
        
        while self.current_token.type != TokenType.RPAREN:
            param_token = self.expect(TokenType.IDENTIFIER)
            param_name = param_token.value
            params.append(param_name)
            
            # Default parameter value
            if self.current_token.type == TokenType.ASSIGN:
                self.advance()
                defaults[param_name] = self.parse_expression()
            
            if self.current_token.type == TokenType.COMMA:
                self.advance()
        
        self.expect(TokenType.RPAREN)
        self.skip_newlines()
        
        # Parse function body
        body = []
        while self.current_token and self.current_token.type != TokenType.END:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
        
        self.expect(TokenType.END)
        
        return FunctionDef(
            name=name,
            params=params,
            body=body,
            defaults=defaults,
            is_async=is_async,
            line=name_token.line,
            column=name_token.column
        )
    
    def parse_class_def(self) -> ClassDef:
        """Parse class definition"""
        self.expect(TokenType.CLASS)
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value
        
        superclass = None
        if self.current_token.type == TokenType.LESS:
            self.advance()
            superclass = self.expect(TokenType.IDENTIFIER).value
        
        self.skip_newlines()
        
        methods = []
        properties = []
        
        while self.current_token and self.current_token.type != TokenType.END:
            if self.current_token.type == TokenType.FUNC:
                methods.append(self.parse_function_def())
            elif self.current_token.type in (TokenType.VAR, TokenType.CONST):
                properties.append(self.parse_var_declaration())
            else:
                self.error(f"Unexpected token in class body: {self.current_token}")
            self.skip_newlines()
        
        self.expect(TokenType.END)
        
        return ClassDef(
            name=name,
            superclass=superclass,
            methods=methods,
            properties=properties,
            line=name_token.line,
            column=name_token.column
        )
    
    def parse_var_declaration(self) -> VarDeclaration:
        """Parse variable declaration"""
        is_const = self.current_token.type == TokenType.CONST
        self.advance()
        
        name_token = self.expect(TokenType.IDENTIFIER)
        name = name_token.value
        
        value = None
        if self.current_token.type == TokenType.ASSIGN:
            self.advance()
            value = self.parse_expression()
        
        return VarDeclaration(
            name=name,
            value=value,
            is_const=is_const,
            line=name_token.line,
            column=name_token.column
        )
    
    def parse_if_statement(self) -> IfStatement:
        """Parse if statement"""
        line = self.current_token.line
        col = self.current_token.column
        
        self.expect(TokenType.IF)
        condition = self.parse_expression()
        
        if self.current_token.type == TokenType.THEN:
            self.advance()
        
        self.skip_newlines()
        
        # Then branch
        then_branch = []
        while self.current_token and self.current_token.type not in (
            TokenType.ELIF, TokenType.ELSE, TokenType.END
        ):
            stmt = self.parse_statement()
            if stmt:
                then_branch.append(stmt)
            self.skip_newlines()
        
        # Elif branches
        elif_branches = []
        while self.current_token and self.current_token.type == TokenType.ELIF:
            self.advance()
            elif_condition = self.parse_expression()
            if self.current_token.type == TokenType.THEN:
                self.advance()
            self.skip_newlines()
            
            elif_body = []
            while self.current_token and self.current_token.type not in (
                TokenType.ELIF, TokenType.ELSE, TokenType.END
            ):
                stmt = self.parse_statement()
                if stmt:
                    elif_body.append(stmt)
                self.skip_newlines()
            
            elif_branches.append((elif_condition, elif_body))
        
        # Else branch
        else_branch = []
        if self.current_token and self.current_token.type == TokenType.ELSE:
            self.advance()
            self.skip_newlines()
            
            while self.current_token and self.current_token.type != TokenType.END:
                stmt = self.parse_statement()
                if stmt:
                    else_branch.append(stmt)
                self.skip_newlines()
        
        self.expect(TokenType.END)
        
        return IfStatement(
            condition=condition,
            then_branch=then_branch,
            elif_branches=elif_branches,
            else_branch=else_branch,
            line=line,
            column=col
        )
    
    def parse_while_loop(self) -> WhileLoop:
        """Parse while loop"""
        line = self.current_token.line
        col = self.current_token.column
        
        self.expect(TokenType.WHILE)
        condition = self.parse_expression()
        
        if self.current_token.type == TokenType.DO:
            self.advance()
        
        self.skip_newlines()
        
        body = []
        while self.current_token and self.current_token.type != TokenType.END:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
        
        self.expect(TokenType.END)
        
        return WhileLoop(condition=condition, body=body, line=line, column=col)
    
    def parse_for_loop(self) -> ForLoop:
        """Parse for loop"""
        line = self.current_token.line
        col = self.current_token.column
        
        self.expect(TokenType.FOR)
        var_token = self.expect(TokenType.IDENTIFIER)
        variable = var_token.value
        
        self.expect(TokenType.IN)
        iterable = self.parse_expression()
        
        if self.current_token.type == TokenType.DO:
            self.advance()
        
        self.skip_newlines()
        
        body = []
        while self.current_token and self.current_token.type != TokenType.END:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
        
        self.expect(TokenType.END)
        
        return ForLoop(variable=variable, iterable=iterable, body=body, line=line, column=col)
    
    def parse_return(self) -> Return:
        """Parse return statement"""
        line = self.current_token.line
        col = self.current_token.column
        
        self.advance()
        
        value = None
        if self.current_token and self.current_token.type not in (
            TokenType.NEWLINE, TokenType.EOF, TokenType.END
        ):
            value = self.parse_expression()
        
        return Return(value=value, line=line, column=col)
    
    def parse_import(self) -> ImportStatement:
        """Parse import statement"""
        line = self.current_token.line
        col = self.current_token.column
        
        items = []
        alias = None
        module = None
        
        # from module import items
        if self.current_token.type == TokenType.FROM:
            self.advance()
            module_token = self.expect(TokenType.IDENTIFIER)
            module = module_token.value
            
            self.expect(TokenType.IMPORT)
            
            # from module import *
            if self.current_token.type == TokenType.MULTIPLY:
                self.advance()
                items = ['*']
            else:
                # from module import item1, item2, item3
                items.append(self.expect(TokenType.IDENTIFIER).value)
                
                while self.current_token and self.current_token.type == TokenType.COMMA:
                    self.advance()
                    items.append(self.expect(TokenType.IDENTIFIER).value)
            
            return ImportStatement(module=module, items=items, alias=None, line=line, column=col)
        
        # import module
        self.expect(TokenType.IMPORT)
        
        # Support for module names with dots and hyphens (e.g., pygame-ce, numpy.random)
        module_parts = []
        module_parts.append(self.expect(TokenType.IDENTIFIER).value)
        
        # Handle dots (numpy.random) or hyphens (pygame-ce)
        while self.current_token and self.current_token.type in (TokenType.DOT, TokenType.MINUS):
            if self.current_token.type == TokenType.DOT:
                module_parts.append('.')
                self.advance()
            elif self.current_token.type == TokenType.MINUS:
                module_parts.append('-')
                self.advance()
            
            if self.current_token.type == TokenType.IDENTIFIER:
                module_parts.append(self.current_token.value)
                self.advance()
            elif self.current_token.type == TokenType.NUMBER:
                # Handle cases like pygame-ce (where 'ce' might be tokenized)
                module_parts.append(str(int(self.current_token.value)))
                self.advance()
        
        module = ''.join(module_parts)
        
        # import module as alias
        if self.current_token and self.current_token.type == TokenType.AS:
            self.advance()
            alias = self.expect(TokenType.IDENTIFIER).value
        
        return ImportStatement(module=module, items=items, alias=alias, line=line, column=col)
    
    def parse_try_statement(self) -> TryStatement:
        """Parse try-catch-finally"""
        line = self.current_token.line
        col = self.current_token.column
        
        self.expect(TokenType.TRY)
        self.skip_newlines()
        
        # Try block
        try_block = []
        while self.current_token and self.current_token.type not in (
            TokenType.CATCH, TokenType.FINALLY, TokenType.END
        ):
            stmt = self.parse_statement()
            if stmt:
                try_block.append(stmt)
            self.skip_newlines()
        
        # Catch block
        catch_var = None
        catch_block = []
        if self.current_token and self.current_token.type == TokenType.CATCH:
            self.advance()
            
            if self.current_token.type == TokenType.IDENTIFIER:
                catch_var = self.advance().value
            
            self.skip_newlines()
            
            while self.current_token and self.current_token.type not in (
                TokenType.FINALLY, TokenType.END
            ):
                stmt = self.parse_statement()
                if stmt:
                    catch_block.append(stmt)
                self.skip_newlines()
        
        # Finally block
        finally_block = []
        if self.current_token and self.current_token.type == TokenType.FINALLY:
            self.advance()
            self.skip_newlines()
            
            while self.current_token and self.current_token.type != TokenType.END:
                stmt = self.parse_statement()
                if stmt:
                    finally_block.append(stmt)
                self.skip_newlines()
        
        self.expect(TokenType.END)
        
        return TryStatement(
            try_block=try_block,
            catch_var=catch_var,
            catch_block=catch_block,
            finally_block=finally_block,
            line=line,
            column=col
        )
    
    def parse_throw(self) -> ThrowStatement:
        """Parse throw statement"""
        line = self.current_token.line
        col = self.current_token.column
        
        self.advance()
        value = self.parse_expression()
        
        return ThrowStatement(value=value, line=line, column=col)
    
    def parse_match_statement(self) -> MatchStatement:
        """Parse match statement"""
        line = self.current_token.line
        col = self.current_token.column
        
        self.expect(TokenType.MATCH)
        value = self.parse_expression()
        self.skip_newlines()
        
        cases = []
        default_case = []
        
        while self.current_token and self.current_token.type != TokenType.END:
            if self.current_token.type == TokenType.CASE:
                self.advance()
                pattern = self.parse_expression()
                self.expect(TokenType.COLON)
                self.skip_newlines()
                
                case_body = []
                while self.current_token and self.current_token.type not in (
                    TokenType.CASE, TokenType.DEFAULT, TokenType.END
                ):
                    stmt = self.parse_statement()
                    if stmt:
                        case_body.append(stmt)
                    self.skip_newlines()
                
                cases.append((pattern, case_body))
            
            elif self.current_token.type == TokenType.DEFAULT:
                self.advance()
                self.expect(TokenType.COLON)
                self.skip_newlines()
                
                while self.current_token and self.current_token.type != TokenType.END:
                    stmt = self.parse_statement()
                    if stmt:
                        default_case.append(stmt)
                    self.skip_newlines()
            else:
                break
        
        self.expect(TokenType.END)
        
        return MatchStatement(
            value=value,
            cases=cases,
            default_case=default_case,
            line=line,
            column=col
        )
    
    def parse_with_statement(self) -> WithStatement:
        """Parse with statement"""
        line = self.current_token.line
        col = self.current_token.column
        
        self.expect(TokenType.WITH)
        context = self.parse_expression()
        
        variable = None
        if self.current_token.type == TokenType.AS:
            self.advance()
            variable = self.expect(TokenType.IDENTIFIER).value
        
        self.skip_newlines()
        
        body = []
        while self.current_token and self.current_token.type != TokenType.END:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
        
        self.expect(TokenType.END)
        
        return WithStatement(
            context=context,
            variable=variable,
            body=body,
            line=line,
            column=col
        )
    
    def parse_expression_statement(self) -> ASTNode:
        """Parse expression as statement"""
        expr = self.parse_expression()
        return expr
    
    def parse_expression(self) -> ASTNode:
        """Parse expression"""
        return self.parse_ternary()
    
    def parse_ternary(self) -> ASTNode:
        """Parse ternary conditional"""
        expr = self.parse_logical_or()
        
        if self.current_token and self.current_token.type == TokenType.QUESTION:
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            
            true_value = self.parse_expression()
            self.expect(TokenType.COLON)
            false_value = self.parse_expression()
            
            return TernaryOp(
                condition=expr,
                true_value=true_value,
                false_value=false_value,
                line=line,
                column=col
            )
        
        return expr
    
    def parse_logical_or(self) -> ASTNode:
        """Parse logical OR"""
        left = self.parse_logical_and()
        
        while self.current_token and self.current_token.type == TokenType.OR:
            op_token = self.advance()
            right = self.parse_logical_and()
            left = BinaryOp(
                left=left,
                operator='or',
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_logical_and(self) -> ASTNode:
        """Parse logical AND"""
        left = self.parse_bitwise_or()
        
        while self.current_token and self.current_token.type == TokenType.AND:
            op_token = self.advance()
            right = self.parse_bitwise_or()
            left = BinaryOp(
                left=left,
                operator='and',
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_bitwise_or(self) -> ASTNode:
        """Parse bitwise OR"""
        left = self.parse_bitwise_xor()
        
        while self.current_token and self.current_token.type == TokenType.BIT_OR:
            op_token = self.advance()
            right = self.parse_bitwise_xor()
            left = BinaryOp(
                left=left,
                operator='|',
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_bitwise_xor(self) -> ASTNode:
        """Parse bitwise XOR"""
        left = self.parse_bitwise_and()
        
        while self.current_token and self.current_token.type == TokenType.BIT_XOR:
            op_token = self.advance()
            right = self.parse_bitwise_and()
            left = BinaryOp(
                left=left,
                operator='^',
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_bitwise_and(self) -> ASTNode:
        """Parse bitwise AND"""
        left = self.parse_equality()
        
        while self.current_token and self.current_token.type == TokenType.BIT_AND:
            op_token = self.advance()
            right = self.parse_equality()
            left = BinaryOp(
                left=left,
                operator='&',
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_equality(self) -> ASTNode:
        """Parse equality operators"""
        left = self.parse_comparison()
        
        while self.current_token and self.current_token.type in (
            TokenType.EQUAL, TokenType.NOT_EQUAL
        ):
            op_token = self.advance()
            op = '==' if op_token.type == TokenType.EQUAL else '!='
            right = self.parse_comparison()
            left = BinaryOp(
                left=left,
                operator=op,
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_comparison(self) -> ASTNode:
        """Parse comparison operators"""
        left = self.parse_bitwise_shift()
        
        while self.current_token and self.current_token.type in (
            TokenType.LESS, TokenType.GREATER,
            TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL
        ):
            op_token = self.advance()
            op_map = {
                TokenType.LESS: '<',
                TokenType.GREATER: '>',
                TokenType.LESS_EQUAL: '<=',
                TokenType.GREATER_EQUAL: '>='
            }
            op = op_map[op_token.type]
            right = self.parse_bitwise_shift()
            left = BinaryOp(
                left=left,
                operator=op,
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_bitwise_shift(self) -> ASTNode:
        """Parse bitwise shift operators"""
        left = self.parse_range()
        
        while self.current_token and self.current_token.type in (
            TokenType.LEFT_SHIFT, TokenType.RIGHT_SHIFT
        ):
            op_token = self.advance()
            op = '<<' if op_token.type == TokenType.LEFT_SHIFT else '>>'
            right = self.parse_range()
            left = BinaryOp(
                left=left,
                operator=op,
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_range(self) -> ASTNode:
        """Parse range operators"""
        left = self.parse_addition()
        
        if self.current_token and self.current_token.type in (
            TokenType.RANGE, TokenType.INCLUSIVE_RANGE
        ):
            inclusive = self.current_token.type == TokenType.INCLUSIVE_RANGE
            line = self.current_token.line
            col = self.current_token.column
            self.advance()
            right = self.parse_addition()
            return RangeExpr(
                start=left,
                end=right,
                inclusive=inclusive,
                line=line,
                column=col
            )
        
        return left
    
    def parse_addition(self) -> ASTNode:
        """Parse addition and subtraction"""
        left = self.parse_multiplication()
        
        while self.current_token and self.current_token.type in (
            TokenType.PLUS, TokenType.MINUS
        ):
            op_token = self.advance()
            op = '+' if op_token.type == TokenType.PLUS else '-'
            right = self.parse_multiplication()
            left = BinaryOp(
                left=left,
                operator=op,
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_multiplication(self) -> ASTNode:
        """Parse multiplication, division, and modulo"""
        left = self.parse_power()
        
        while self.current_token and self.current_token.type in (
            TokenType.MULTIPLY, TokenType.DIVIDE,
            TokenType.MODULO, TokenType.FLOOR_DIV
        ):
            op_token = self.advance()
            op_map = {
                TokenType.MULTIPLY: '*',
                TokenType.DIVIDE: '/',
                TokenType.MODULO: '%',
                TokenType.FLOOR_DIV: '//'
            }
            op = op_map[op_token.type]
            right = self.parse_power()
            left = BinaryOp(
                left=left,
                operator=op,
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_power(self) -> ASTNode:
        """Parse power operator"""
        left = self.parse_unary()
        
        if self.current_token and self.current_token.type == TokenType.POWER:
            op_token = self.advance()
            right = self.parse_power()  # Right associative
            return BinaryOp(
                left=left,
                operator='**',
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_unary(self) -> ASTNode:
        """Parse unary operators"""
        if self.current_token and self.current_token.type in (
            TokenType.NOT, TokenType.MINUS, TokenType.BIT_NOT
        ):
            op_token = self.advance()
            op_map = {
                TokenType.NOT: 'not',
                TokenType.MINUS: '-',
                TokenType.BIT_NOT: '~'
            }
            op = op_map[op_token.type]
            operand = self.parse_unary()
            return UnaryOp(
                operator=op,
                operand=operand,
                line=op_token.line,
                column=op_token.column
            )
        
        return self.parse_postfix()
    
    def parse_postfix(self) -> ASTNode:
        """Parse postfix expressions"""
        expr = self.parse_primary()
        
        while True:
            if not self.current_token:
                break
            
            # Function call
            if self.current_token.type == TokenType.LPAREN:
                line = self.current_token.line
                col = self.current_token.column
                self.advance()
                
                arguments = []
                while self.current_token.type != TokenType.RPAREN:
                    arguments.append(self.parse_expression())
                    if self.current_token.type == TokenType.COMMA:
                        self.advance()
                
                self.expect(TokenType.RPAREN)
                expr = FunctionCall(
                    function=expr,
                    arguments=arguments,
                    line=line,
                    column=col
                )
            
            # Index access
            elif self.current_token.type == TokenType.LBRACKET:
                line = self.current_token.line
                col = self.current_token.column
                self.advance()
                index = self.parse_expression()
                self.expect(TokenType.RBRACKET)
                expr = IndexAccess(
                    object=expr,
                    index=index,
                    line=line,
                    column=col
                )
            
            # Member access
            elif self.current_token.type == TokenType.DOT:
                line = self.current_token.line
                col = self.current_token.column
                self.advance()
                member = self.expect(TokenType.IDENTIFIER).value
                expr = MemberAccess(
                    object=expr,
                    member=member,
                    line=line,
                    column=col
                )
            
            # Assignment operators
            elif self.current_token.type in (
                TokenType.ASSIGN, TokenType.PLUS_ASSIGN,
                TokenType.MINUS_ASSIGN, TokenType.MULTIPLY_ASSIGN,
                TokenType.DIVIDE_ASSIGN, TokenType.MODULO_ASSIGN
            ):
                op_token = self.advance()
                op_map = {
                    TokenType.ASSIGN: '=',
                    TokenType.PLUS_ASSIGN: '+=',
                    TokenType.MINUS_ASSIGN: '-=',
                    TokenType.MULTIPLY_ASSIGN: '*=',
                    TokenType.DIVIDE_ASSIGN: '/=',
                    TokenType.MODULO_ASSIGN: '%='
                }
                op = op_map[op_token.type]
                value = self.parse_expression()
                expr = Assignment(
                    target=expr,
                    value=value,
                    operator=op,
                    line=op_token.line,
                    column=op_token.column
                )
            else:
                break
        
        return expr
    
    def parse_primary(self) -> ASTNode:
        """Parse primary expressions"""
        token = self.current_token
        
        if not token:
            self.error("Unexpected end of input")
        
        # Number literal
        if token.type == TokenType.NUMBER:
            self.advance()
            return NumberLiteral(value=token.value, line=token.line, column=token.column)
        
        # String literal
        if token.type == TokenType.STRING:
            self.advance()
            return StringLiteral(value=token.value, line=token.line, column=token.column)
        
        # Boolean literal
        if token.type == TokenType.BOOLEAN:
            self.advance()
            return BooleanLiteral(value=token.value, line=token.line, column=token.column)
        
        # Null literal
        if token.type == TokenType.NULL:
            self.advance()
            return NullLiteral(line=token.line, column=token.column)
        
        # This
        if token.type == TokenType.THIS:
            self.advance()
            return This(line=token.line, column=token.column)
        
        # Super
        if token.type == TokenType.SUPER:
            self.advance()
            return Super(line=token.line, column=token.column)
        
        # New instance
        if token.type == TokenType.NEW:
            self.advance()
            class_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.LPAREN)
            
            arguments = []
            while self.current_token.type != TokenType.RPAREN:
                arguments.append(self.parse_expression())
                if self.current_token.type == TokenType.COMMA:
                    self.advance()
            
            self.expect(TokenType.RPAREN)
            return NewInstance(
                class_name=class_name,
                arguments=arguments,
                line=token.line,
                column=token.column
            )
        
        # Await
        if token.type == TokenType.AWAIT:
            self.advance()
            expr = self.parse_postfix()
            if isinstance(expr, FunctionCall):
                expr.is_await = True
                return expr
            self.error("await must be used with function call")
        
        # Lambda
        if token.type == TokenType.LAMBDA:
            return self.parse_lambda()
        
        # Identifier
        if token.type == TokenType.IDENTIFIER:
            self.advance()
            return Identifier(name=token.value, line=token.line, column=token.column)
        
        # Parenthesized expression
        if token.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        # Array literal
        if token.type == TokenType.LBRACKET:
            return self.parse_array_literal()
        
        # Dict literal
        if token.type == TokenType.LBRACE:
            return self.parse_dict_literal()
        
        self.error(f"Unexpected token: {token}")
    
    def parse_lambda(self) -> LambdaExpr:
        """Parse lambda expression"""
        line = self.current_token.line
        col = self.current_token.column
        
        self.expect(TokenType.LAMBDA)
        
        params = []
        if self.current_token.type == TokenType.IDENTIFIER:
            params.append(self.advance().value)
            
            while self.current_token.type == TokenType.COMMA:
                self.advance()
                params.append(self.expect(TokenType.IDENTIFIER).value)
        
        self.expect(TokenType.COLON)
        body = self.parse_expression()
        
        return LambdaExpr(params=params, body=body, line=line, column=col)
    
    def parse_array_literal(self) -> ArrayLiteral:
        """Parse array literal"""
        line = self.current_token.line
        col = self.current_token.column
        
        self.expect(TokenType.LBRACKET)
        
        elements = []
        while self.current_token.type != TokenType.RBRACKET:
            elements.append(self.parse_expression())
            if self.current_token.type == TokenType.COMMA:
                self.advance()
        
        self.expect(TokenType.RBRACKET)
        
        return ArrayLiteral(elements=elements, line=line, column=col)
    
    def parse_dict_literal(self) -> DictLiteral:
        """Parse dictionary literal"""
        line = self.current_token.line
        col = self.current_token.column
        
        self.expect(TokenType.LBRACE)
        
        pairs = []
        while self.current_token.type != TokenType.RBRACE:
            key = self.parse_expression()
            self.expect(TokenType.COLON)
            value = self.parse_expression()
            pairs.append((key, value))
            
            if self.current_token.type == TokenType.COMMA:
                self.advance()
        
        self.expect(TokenType.RBRACE)
        
        return DictLiteral(pairs=pairs, line=line, column=col)


# ============================================================================
# RUNTIME AND INTERPRETER
# ============================================================================

class BreakException(Exception):
    """Break statement exception"""
    pass


class ContinueException(Exception):
    """Continue statement exception"""
    pass


class ReturnException(Exception):
    """Return statement exception"""
    def __init__(self, value):
        self.value = value
        super().__init__()


class HubbleException(Exception):
    """Hubble runtime exception"""
    pass


@dataclass
class Environment:
    """Variable environment/scope"""
    parent: Optional['Environment'] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    constants: set = field(default_factory=set)
    
    def define(self, name: str, value: Any, is_const: bool = False):
        """Define variable"""
        if name in self.constants:
            raise HubbleException(f"Cannot reassign constant: {name}")
        self.variables[name] = value
        if is_const:
            self.constants.add(name)
    
    def get(self, name: str) -> Any:
        """Get variable value"""
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.get(name)
        raise HubbleException(f"Undefined variable: {name}")
    
    def set(self, name: str, value: Any):
        """Set variable value"""
        if name in self.constants:
            raise HubbleException(f"Cannot reassign constant: {name}")
        if name in self.variables:
            self.variables[name] = value
            return
        if self.parent:
            self.parent.set(name, value)
            return
        raise HubbleException(f"Undefined variable: {name}")
    
    def has(self, name: str) -> bool:
        """Check if variable exists"""
        if name in self.variables:
            return True
        if self.parent:
            return self.parent.has(name)
        return False


class HubbleFunction:
    """Callable function"""
    def __init__(self, name: str, params: List[str], body: List[ASTNode], 
                 closure: Environment, defaults: Dict[str, Any] = None,
                 is_async: bool = False):
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure
        self.defaults = defaults or {}
        self.is_async = is_async
    
    def __repr__(self):
        return f"<function {self.name}>"


class HubbleClass:
    """Class definition"""
    def __init__(self, name: str, methods: Dict[str, HubbleFunction],
                 properties: Dict[str, Any], superclass: Optional['HubbleClass'] = None):
        self.name = name
        self.methods = methods
        self.properties = properties
        self.superclass = superclass
    
    def __repr__(self):
        return f"<class {self.name}>"


class HubbleInstance:
    """Class instance"""
    def __init__(self, klass: HubbleClass):
        self.klass = klass
        self.fields = dict(klass.properties)
    
    def get(self, name: str) -> Any:
        """Get instance field or method"""
        if name in self.fields:
            return self.fields[name]
        
        method = self.klass.methods.get(name)
        if method:
            return self.bind_method(method)
        
        if self.klass.superclass:
            method = self.klass.superclass.methods.get(name)
            if method:
                return self.bind_method(method)
        
        raise HubbleException(f"Undefined property: {name}")
    
    def set(self, name: str, value: Any):
        """Set instance field"""
        self.fields[name] = value
    
    def bind_method(self, method: HubbleFunction):
        """Bind method to instance"""
        def bound(*args, **kwargs):
            from types import SimpleNamespace
            this_obj = SimpleNamespace(
                _instance=self,
                __getattr__=lambda s, n: self.get(n),
                __setattr__=lambda s, n, v: self.set(n, v) if n != '_instance' else object.__setattr__(s, n, v)
            )
            return Interpreter.call_function(method, args, this_obj)
        return bound
    
    def __repr__(self):
        return f"<{self.klass.name} instance>"


class Interpreter:
    """Hubble interpreter"""
    
    def __init__(self):
        self.global_env = Environment()
        self.current_env = self.global_env
        self.module_manager = ModuleManager()
        self.module_cache = {}  # Cache para prevenir imports circulares
        self.setup_builtins()
    
    def setup_builtins(self):
        """Setup built-in functions and variables"""
        
        # Print function
        def builtin_print(*args):
            print(*args)
            return None
        
        # Input function
        def builtin_input(prompt=""):
            return input(prompt)
        
        # Type function
        def builtin_type(obj):
            type_map = {
                int: 'number',
                float: 'number',
                str: 'string',
                bool: 'boolean',
                list: 'array',
                dict: 'dict',
                type(None): 'null'
            }
            return type_map.get(type(obj), 'object')
        
        # String functions
        def builtin_str(obj):
            return str(obj)
        
        def builtin_int(obj):
            return int(obj)
        
        def builtin_float(obj):
            return float(obj)
        
        # Array functions
        def builtin_len(obj):
            return len(obj)
        
        def builtin_push(arr, item):
            if isinstance(arr, list):
                arr.append(item)
                return arr
            raise HubbleException("push() requires array")
        
        def builtin_pop(arr):
            if isinstance(arr, list):
                return arr.pop() if arr else None
            raise HubbleException("pop() requires array")
        
        def builtin_shift(arr):
            if isinstance(arr, list):
                return arr.pop(0) if arr else None
            raise HubbleException("shift() requires array")
        
        def builtin_unshift(arr, item):
            if isinstance(arr, list):
                arr.insert(0, item)
                return arr
            raise HubbleException("unshift() requires array")
        
        def builtin_slice(arr, start, end=None):
            if end is None:
                return arr[start:]
            return arr[start:end]
        
        def builtin_join(arr, sep=""):
            return sep.join(str(x) for x in arr)
        
        def builtin_split(s, sep=None):
            return s.split(sep)
        
        def builtin_reverse(arr):
            if isinstance(arr, list):
                arr.reverse()
                return arr
            raise HubbleException("reverse() requires array")
        
        def builtin_sort(arr, key=None, reverse=False):
            if isinstance(arr, list):
                arr.sort(key=key, reverse=reverse)
                return arr
            raise HubbleException("sort() requires array")
        
        # Math functions
        def builtin_abs(x):
            return abs(x)
        
        def builtin_min(*args):
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                return min(args[0])
            return min(args)
        
        def builtin_max(*args):
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                return max(args[0])
            return max(args)
        
        def builtin_sum(arr):
            return sum(arr)
        
        def builtin_sqrt(x):
            return math.sqrt(x)
        
        def builtin_pow(x, y):
            return math.pow(x, y)
        
        def builtin_floor(x):
            return math.floor(x)
        
        def builtin_ceil(x):
            return math.ceil(x)
        
        def builtin_round(x, n=0):
            return round(x, n)
        
        def builtin_sin(x):
            return math.sin(x)
        
        def builtin_cos(x):
            return math.cos(x)
        
        def builtin_tan(x):
            return math.tan(x)
        
        def builtin_log(x, base=math.e):
            return math.log(x, base)
        
        def builtin_exp(x):
            return math.exp(x)
        
        # Random functions
        def builtin_random():
            return random.random()
        
        def builtin_randint(a, b):
            return random.randint(a, b)
        
        def builtin_choice(arr):
            return random.choice(arr)
        
        def builtin_shuffle(arr):
            random.shuffle(arr)
            return arr
        
        # Utility functions
        def builtin_range(*args):
            return list(range(*args))
        
        def builtin_map(func, iterable):
            return [func(x) for x in iterable]
        
        def builtin_filter(func, iterable):
            return [x for x in iterable if func(x)]
        
        def builtin_reduce(func, iterable, initial=None):
            from functools import reduce as py_reduce
            if initial is not None:
                return py_reduce(func, iterable, initial)
            return py_reduce(func, iterable)
        
        def builtin_zip(*iterables):
            return list(zip(*iterables))
        
        def builtin_enumerate(iterable, start=0):
            return list(enumerate(iterable, start))
        
        def builtin_all(iterable):
            return all(iterable)
        
        def builtin_any(iterable):
            return any(iterable)
        
        # Time functions
        def builtin_time():
            return time.time()
        
        def builtin_sleep(seconds):
            time.sleep(seconds)
            return None
        
        # File I/O
        def builtin_read_file(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        
        def builtin_write_file(path, content):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return None
        
        def builtin_append_file(path, content):
            with open(path, 'a', encoding='utf-8') as f:
                f.write(content)
            return None
        
        def builtin_file_exists(path):
            return os.path.exists(path)
        
        # JSON functions
        def builtin_json_parse(s):
            return json.loads(s)
        
        def builtin_json_stringify(obj, indent=None):
            return json.dumps(obj, indent=indent)
        
        # String functions
        def builtin_upper(s):
            return s.upper()
        
        def builtin_lower(s):
            return s.lower()
        
        def builtin_trim(s):
            return s.strip()
        
        def builtin_replace(s, old, new):
            return s.replace(old, new)
        
        def builtin_startswith(s, prefix):
            return s.startswith(prefix)
        
        def builtin_endswith(s, suffix):
            return s.endswith(suffix)
        
        def builtin_contains(s, substr):
            return substr in s
        
        def builtin_index_of(s, substr):
            try:
                return s.index(substr)
            except ValueError:
                return -1
        
        # Advanced functions
        def builtin_keys(d):
            return list(d.keys())
        
        def builtin_values(d):
            return list(d.values())
        
        def builtin_items(d):
            return list(d.items())
        
        def builtin_has_key(d, key):
            return key in d
        
        def builtin_delete(d, key):
            if key in d:
                del d[key]
            return d
        
        def builtin_clone(obj):
            return copy.deepcopy(obj)
        
        def builtin_is_null(obj):
            return obj is None
        
        def builtin_is_number(obj):
            return isinstance(obj, (int, float))
        
        def builtin_is_string(obj):
            return isinstance(obj, str)
        
        def builtin_is_boolean(obj):
            return isinstance(obj, bool)
        
        def builtin_is_array(obj):
            return isinstance(obj, list)
        
        def builtin_is_dict(obj):
            return isinstance(obj, dict)
        
        def builtin_exit(code=0):
            sys.exit(code)
        
        def builtin_assert(condition, message="Assertion failed"):
            if not condition:
                raise HubbleException(message)
            return True
        
        # Register all builtins
        builtins = {
            'print': builtin_print,
            'input': builtin_input,
            'type': builtin_type,
            'str': builtin_str,
            'int': builtin_int,
            'float': builtin_float,
            'len': builtin_len,
            'push': builtin_push,
            'pop': builtin_pop,
            'shift': builtin_shift,
            'unshift': builtin_unshift,
            'slice': builtin_slice,
            'join': builtin_join,
            'split': builtin_split,
            'reverse': builtin_reverse,
            'sort': builtin_sort,
            'abs': builtin_abs,
            'min': builtin_min,
            'max': builtin_max,
            'sum': builtin_sum,
            'sqrt': builtin_sqrt,
            'pow': builtin_pow,
            'floor': builtin_floor,
            'ceil': builtin_ceil,
            'round': builtin_round,
            'sin': builtin_sin,
            'cos': builtin_cos,
            'tan': builtin_tan,
            'log': builtin_log,
            'exp': builtin_exp,
            'random': builtin_random,
            'randint': builtin_randint,
            'choice': builtin_choice,
            'shuffle': builtin_shuffle,
            'range': builtin_range,
            'map': builtin_map,
            'filter': builtin_filter,
            'reduce': builtin_reduce,
            'zip': builtin_zip,
            'enumerate': builtin_enumerate,
            'all': builtin_all,
            'any': builtin_any,
            'time': builtin_time,
            'sleep': builtin_sleep,
            'read_file': builtin_read_file,
            'write_file': builtin_write_file,
            'append_file': builtin_append_file,
            'file_exists': builtin_file_exists,
            'json_parse': builtin_json_parse,
            'json_stringify': builtin_json_stringify,
            'upper': builtin_upper,
            'lower': builtin_lower,
            'trim': builtin_trim,
            'replace': builtin_replace,
            'startswith': builtin_startswith,
            'endswith': builtin_endswith,
            'contains': builtin_contains,
            'index_of': builtin_index_of,
            'keys': builtin_keys,
            'values': builtin_values,
            'items': builtin_items,
            'has_key': builtin_has_key,
            'delete': builtin_delete,
            'clone': builtin_clone,
            'is_null': builtin_is_null,
            'is_number': builtin_is_number,
            'is_string': builtin_is_string,
            'is_boolean': builtin_is_boolean,
            'is_array': builtin_is_array,
            'is_dict': builtin_is_dict,
            'exit': builtin_exit,
            'assert': builtin_assert,
        }
        
        # Constants
        self.global_env.define('PI', math.pi, is_const=True)
        self.global_env.define('E', math.e, is_const=True)
        self.global_env.define('null', None, is_const=True)
        self.global_env.define('true', True, is_const=True)
        self.global_env.define('false', False, is_const=True)
        
        for name, func in builtins.items():
            self.global_env.define(name, func)
    
    def interpret(self, program: Program):
        """Interpret program"""
        try:
            for statement in program.statements:
                self.eval_statement(statement)
        except (BreakException, ContinueException):
            raise HubbleException("break/continue outside loop")
        except ReturnException:
            raise HubbleException("return outside function")
    
    def eval_statement(self, node: ASTNode) -> Any:
        """Evaluate statement"""
        if isinstance(node, VarDeclaration):
            return self.eval_var_declaration(node)
        elif isinstance(node, FunctionDef):
            return self.eval_function_def(node)
        elif isinstance(node, ClassDef):
            return self.eval_class_def(node)
        elif isinstance(node, IfStatement):
            return self.eval_if_statement(node)
        elif isinstance(node, WhileLoop):
            return self.eval_while_loop(node)
        elif isinstance(node, ForLoop):
            return self.eval_for_loop(node)
        elif isinstance(node, Return):
            return self.eval_return(node)
        elif isinstance(node, Break):
            raise BreakException()
        elif isinstance(node, Continue):
            raise ContinueException()
        elif isinstance(node, ImportStatement):
            return self.eval_import(node)
        elif isinstance(node, TryStatement):
            return self.eval_try_statement(node)
        elif isinstance(node, ThrowStatement):
            return self.eval_throw(node)
        elif isinstance(node, MatchStatement):
            return self.eval_match_statement(node)
        elif isinstance(node, WithStatement):
            return self.eval_with_statement(node)
        else:
            return self.eval_expression(node)
    
    def eval_expression(self, node: ASTNode) -> Any:
        """Evaluate expression"""
        if isinstance(node, NumberLiteral):
            return node.value
        elif isinstance(node, StringLiteral):
            return node.value
        elif isinstance(node, BooleanLiteral):
            return node.value
        elif isinstance(node, NullLiteral):
            return None
        elif isinstance(node, Identifier):
            return self.current_env.get(node.name)
        elif isinstance(node, BinaryOp):
            return self.eval_binary_op(node)
        elif isinstance(node, UnaryOp):
            return self.eval_unary_op(node)
        elif isinstance(node, Assignment):
            return self.eval_assignment(node)
        elif isinstance(node, FunctionCall):
            return self.eval_function_call(node)
        elif isinstance(node, ArrayLiteral):
            return [self.eval_expression(e) for e in node.elements]
        elif isinstance(node, DictLiteral):
            return {
                self.eval_expression(k): self.eval_expression(v)
                for k, v in node.pairs
            }
        elif isinstance(node, IndexAccess):
            return self.eval_index_access(node)
        elif isinstance(node, MemberAccess):
            return self.eval_member_access(node)
        elif isinstance(node, NewInstance):
            return self.eval_new_instance(node)
        elif isinstance(node, This):
            return self.current_env.get('this')
        elif isinstance(node, Super):
            return self.current_env.get('super')
        elif isinstance(node, LambdaExpr):
            return self.eval_lambda(node)
        elif isinstance(node, RangeExpr):
            return self.eval_range(node)
        elif isinstance(node, TernaryOp):
            return self.eval_ternary(node)
        else:
            raise HubbleException(f"Unknown expression type: {type(node)}")
    
    def eval_var_declaration(self, node: VarDeclaration):
        """Evaluate variable declaration"""
        value = None if node.value is None else self.eval_expression(node.value)
        self.current_env.define(node.name, value, node.is_const)
        return None
    
    def eval_function_def(self, node: FunctionDef):
        """Evaluate function definition"""
        defaults = {k: self.eval_expression(v) for k, v in node.defaults.items()}
        func = HubbleFunction(
            node.name,
            node.params,
            node.body,
            self.current_env,
            defaults,
            node.is_async
        )
        self.current_env.define(node.name, func)
        return None
    
    def eval_class_def(self, node: ClassDef):
        """Evaluate class definition"""
        superclass = None
        if node.superclass:
            superclass_val = self.current_env.get(node.superclass)
            if not isinstance(superclass_val, HubbleClass):
                raise HubbleException(f"Superclass must be a class: {node.superclass}")
            superclass = superclass_val
        
        methods = {}
        for method_node in node.methods:
            defaults = {k: self.eval_expression(v) for k, v in method_node.defaults.items()}
            methods[method_node.name] = HubbleFunction(
                method_node.name,
                method_node.params,
                method_node.body,
                self.current_env,
                defaults,
                method_node.is_async
            )
        
        properties = {}
        for prop in node.properties:
            value = None if prop.value is None else self.eval_expression(prop.value)
            properties[prop.name] = value
        
        klass = HubbleClass(node.name, methods, properties, superclass)
        self.current_env.define(node.name, klass)
        return None
    
    def eval_binary_op(self, node: BinaryOp) -> Any:
        """Evaluate binary operation"""
        left = self.eval_expression(node.left)
        
        # Short-circuit evaluation
        if node.operator == 'and':
            return left and self.eval_expression(node.right)
        elif node.operator == 'or':
            return left or self.eval_expression(node.right)
        
        right = self.eval_expression(node.right)
        
        operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '//': operator.floordiv,
            '%': operator.mod,
            '**': operator.pow,
            '==': operator.eq,
            '!=': operator.ne,
            '<': operator.lt,
            '>': operator.gt,
            '<=': operator.le,
            '>=': operator.ge,
            '&': operator.and_,
            '|': operator.or_,
            '^': operator.xor,
            '<<': operator.lshift,
            '>>': operator.rshift,
        }
        
        if node.operator in operators:
            try:
                return operators[node.operator](left, right)
            except Exception as e:
                raise HubbleException(f"Operation error: {e}")
        
        raise HubbleException(f"Unknown operator: {node.operator}")
    
    def eval_unary_op(self, node: UnaryOp) -> Any:
        """Evaluate unary operation"""
        operand = self.eval_expression(node.operand)
        
        if node.operator == 'not':
            return not operand
        elif node.operator == '-':
            return -operand
        elif node.operator == '~':
            return ~int(operand)
        
        raise HubbleException(f"Unknown unary operator: {node.operator}")
    
    def eval_assignment(self, node: Assignment) -> Any:
        """Evaluate assignment"""
        value = self.eval_expression(node.value)
        
        if isinstance(node.target, Identifier):
            if node.operator == '=':
                self.current_env.set(node.target.name, value)
            elif node.operator == '+=':
                current = self.current_env.get(node.target.name)
                self.current_env.set(node.target.name, current + value)
            elif node.operator == '-=':
                current = self.current_env.get(node.target.name)
                self.current_env.set(node.target.name, current - value)
            elif node.operator == '*=':
                current = self.current_env.get(node.target.name)
                self.current_env.set(node.target.name, current * value)
            elif node.operator == '/=':
                current = self.current_env.get(node.target.name)
                self.current_env.set(node.target.name, current / value)
            elif node.operator == '%=':
                current = self.current_env.get(node.target.name)
                self.current_env.set(node.target.name, current % value)
        
        elif isinstance(node.target, IndexAccess):
            obj = self.eval_expression(node.target.object)
            index = self.eval_expression(node.target.index)
            
            if node.operator == '=':
                obj[index] = value
            elif node.operator == '+=':
                obj[index] += value
            elif node.operator == '-=':
                obj[index] -= value
            elif node.operator == '*=':
                obj[index] *= value
            elif node.operator == '/=':
                obj[index] /= value
            elif node.operator == '%=':
                obj[index] %= value
        
        elif isinstance(node.target, MemberAccess):
            obj = self.eval_expression(node.target.object)
            if isinstance(obj, HubbleInstance):
                if node.operator == '=':
                    obj.set(node.target.member, value)
                else:
                    current = obj.get(node.target.member)
                    if node.operator == '+=':
                        obj.set(node.target.member, current + value)
                    elif node.operator == '-=':
                        obj.set(node.target.member, current - value)
                    elif node.operator == '*=':
                        obj.set(node.target.member, current * value)
                    elif node.operator == '/=':
                        obj.set(node.target.member, current / value)
                    elif node.operator == '%=':
                        obj.set(node.target.member, current % value)
        
        return value
    
    def eval_function_call(self, node: FunctionCall) -> Any:
        """Evaluate function call"""
        func = self.eval_expression(node.function)
        args = [self.eval_expression(arg) for arg in node.arguments]
        
        # Built-in function
        if callable(func) and not isinstance(func, HubbleFunction):
            try:
                return func(*args)
            except Exception as e:
                raise HubbleException(f"Function call error: {e}")
        
        # User-defined function
        if isinstance(func, HubbleFunction):
            return self.call_function(func, args)
        
        raise HubbleException(f"Not a function: {func}")
    
    @staticmethod
    def call_function(func: HubbleFunction, args: List[Any], this_obj=None) -> Any:
        """Call Hubble function"""
        if len(args) > len(func.params):
            raise HubbleException(f"Too many arguments for {func.name}")
        
        # Create new environment
        func_env = Environment(parent=func.closure)
        
        # Bind parameters
        for i, param in enumerate(func.params):
            if i < len(args):
                func_env.define(param, args[i])
            elif param in func.defaults:
                func_env.define(param, func.defaults[param])
            else:
                raise HubbleException(f"Missing required argument: {param}")
        
        # Bind this if provided
        if this_obj is not None:
            func_env.define('this', this_obj)
        
        # Execute function body
        interpreter = Interpreter()
        interpreter.current_env = func_env
        interpreter.global_env = func.closure
        
        try:
            for stmt in func.body:
                interpreter.eval_statement(stmt)
        except ReturnException as e:
            return e.value
        
        return None
    
    def eval_if_statement(self, node: IfStatement):
        """Evaluate if statement"""
        condition = self.eval_expression(node.condition)
        
        if condition:
            for stmt in node.then_branch:
                self.eval_statement(stmt)
        else:
            # Check elif branches
            for elif_condition, elif_body in node.elif_branches:
                if self.eval_expression(elif_condition):
                    for stmt in elif_body:
                        self.eval_statement(stmt)
                    return
            
            # Else branch
            for stmt in node.else_branch:
                self.eval_statement(stmt)
    
    def eval_while_loop(self, node: WhileLoop):
        """Evaluate while loop"""
        while self.eval_expression(node.condition):
            try:
                for stmt in node.body:
                    self.eval_statement(stmt)
            except BreakException:
                break
            except ContinueException:
                continue
    
    def eval_for_loop(self, node: ForLoop):
        """Evaluate for loop"""
        iterable = self.eval_expression(node.iterable)
        
        if not hasattr(iterable, '__iter__'):
            raise HubbleException("for loop requires iterable")
        
        loop_env = Environment(parent=self.current_env)
        prev_env = self.current_env
        self.current_env = loop_env
        
        try:
            for item in iterable:
                loop_env.define(node.variable, item)
                try:
                    for stmt in node.body:
                        self.eval_statement(stmt)
                except BreakException:
                    break
                except ContinueException:
                    continue
        finally:
            self.current_env = prev_env
    
    def eval_return(self, node: Return):
        """Evaluate return statement"""
        value = None if node.value is None else self.eval_expression(node.value)
        raise ReturnException(value)
    
    def eval_index_access(self, node: IndexAccess) -> Any:
        """Evaluate index access"""
        obj = self.eval_expression(node.object)
        index = self.eval_expression(node.index)
        
        try:
            return obj[index]
        except (KeyError, IndexError, TypeError) as e:
            raise HubbleException(f"Index access error: {e}")
    
    def eval_member_access(self, node: MemberAccess) -> Any:
        """Evaluate member access"""
        obj = self.eval_expression(node.object)
        
        if isinstance(obj, HubbleInstance):
            return obj.get(node.member)
        elif isinstance(obj, dict):
            return obj.get(node.member)
        else:
            raise HubbleException(f"Cannot access member '{node.member}' on {type(obj)}")
    
    def eval_new_instance(self, node: NewInstance) -> Any:
        """Evaluate new instance creation"""
        klass = self.current_env.get(node.class_name)
        
        if not isinstance(klass, HubbleClass):
            raise HubbleException(f"Not a class: {node.class_name}")
        
        instance = HubbleInstance(klass)
        
        # Call constructor if exists
        if 'init' in klass.methods:
            constructor = klass.methods['init']
            args = [self.eval_expression(arg) for arg in node.arguments]
            self.call_function(constructor, args, instance)
        
        return instance
    
    def eval_lambda(self, node: LambdaExpr) -> HubbleFunction:
        """Evaluate lambda expression"""
        return HubbleFunction(
            '<lambda>',
            node.params,
            [Return(value=node.body)],
            self.current_env
        )
    
    def eval_range(self, node: RangeExpr) -> list:
        """Evaluate range expression"""
        start = self.eval_expression(node.start)
        end = self.eval_expression(node.end)
        
        if not isinstance(start, int) or not isinstance(end, int):
            raise HubbleException("Range requires integer values")
        
        if node.inclusive:
            return list(range(start, end + 1))
        else:
            return list(range(start, end))
    
    def eval_ternary(self, node: TernaryOp) -> Any:
        """Evaluate ternary conditional"""
        condition = self.eval_expression(node.condition)
        if condition:
            return self.eval_expression(node.true_value)
        else:
            return self.eval_expression(node.false_value)
    
    def eval_import(self, node: ImportStatement):
        """Evaluate import statement with full support for .hbl, .py, .exe, .dll, .so, .dylib, and Python packages"""
        module_name = node.module
        
        # Check if it's a standard library module first
        if module_name in self.module_manager.stdlib_modules:
            module_obj = self.module_manager.stdlib_modules[module_name]
            
            # from module import items
            if node.items:
                if node.items == ['*']:
                    # from module import * - import all
                    for name, value in module_obj.items():
                        if not name.startswith('_'):
                            self.current_env.define(name, value)
                else:
                    # from module import item1, item2
                    for item in node.items:
                        if item in module_obj:
                            self.current_env.define(item, module_obj[item])
                        else:
                            raise HubbleException(f"Cannot import '{item}' from module '{module_name}'")
            else:
                # import module or import module as alias
                alias = node.alias or module_name
                self.current_env.define(alias, module_obj)
            
            return None
        
        # Try to find module file (.hbl, .py, .exe, .dll, .so, .dylib)
        module_path, module_type = self.find_module_with_type(module_name)
        
        # If not found as file, try to import as Python package (pip installed)
        if not module_path:
            try:
                module_obj = self.load_python_package(module_name)
                
                # from module import items
                if node.items:
                    if node.items == ['*']:
                        # from module import * - import all non-private items
                        for name, value in module_obj.items():
                            if not name.startswith('_'):
                                self.current_env.define(name, value)
                    else:
                        # from module import item1, item2, item3
                        for item in node.items:
                            if item in module_obj:
                                self.current_env.define(item, module_obj[item])
                            else:
                                # Try to get from the actual module
                                if '__module__' in module_obj:
                                    try:
                                        value = getattr(module_obj['__module__'], item)
                                        self.current_env.define(item, value)
                                    except AttributeError:
                                        raise HubbleException(f"Cannot import '{item}' from module '{module_name}'")
                                else:
                                    raise HubbleException(f"Cannot import '{item}' from module '{module_name}'")
                else:
                    # import module or import module as alias
                    alias = node.alias or module_name.replace('-', '_').replace('.', '_')
                    self.current_env.define(alias, module_obj)
                
                return None
            except HubbleException:
                raise  # Re-raise Hubble exceptions
            except Exception as e:
                raise HubbleException(f"Cannot find module: {module_name}. Error: {e}")
        
        # Load module based on type
        if module_type == 'hbl':
            module_obj = self.load_hubble_module(module_path, module_name)
        elif module_type == 'py':
            module_obj = self.load_python_module(module_path, module_name)
        elif module_type in ['dll', 'so', 'dylib', 'pyd']:
            # Native libraries use the old native loader
            module_obj = self.load_native_module(module_path, module_name, module_type)
        else:
            # Use the universal generic loader for everything else
            module_obj = self.load_generic_module(module_path, module_name, module_type)
        
        # from module import items
        if node.items:
            if node.items == ['*']:
                # from module import * - import all non-private items
                for name, value in module_obj.items():
                    if not name.startswith('_'):
                        self.current_env.define(name, value)
            else:
                # from module import item1, item2, item3
                for item in node.items:
                    if item in module_obj:
                        self.current_env.define(item, module_obj[item])
                    else:
                        raise HubbleException(f"Cannot import '{item}' from module '{module_name}'")
        else:
            # import module or import module as alias
            alias = node.alias or module_name.replace('-', '_').replace('.', '_')
            self.current_env.define(alias, module_obj)
        
        return None
    
    def load_hubble_module(self, module_path: str, module_name: str) -> dict:
        """Load .hbl Hubble module with circular import detection"""
        # Normalizar o caminho do módulo
        normalized_path = str(Path(module_path).resolve())
        
        # Verificar se o módulo já está no cache
        if normalized_path in self.module_cache:
            return self.module_cache[normalized_path]
        
        # Marcar módulo como "sendo carregado" para detectar imports circulares
        if normalized_path in getattr(self, '_loading_modules', set()):
            raise HubbleException(f"Circular import detected: {module_name}")
        
        # Inicializar set de módulos sendo carregados se não existir
        if not hasattr(self, '_loading_modules'):
            self._loading_modules = set()
        
        self._loading_modules.add(normalized_path)
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Create module environment
            module_env = Environment(parent=self.global_env)
            
            # Parse and execute module
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            program = parser.parse()
            
            interpreter = Interpreter()
            interpreter.current_env = module_env
            interpreter.global_env = self.global_env
            interpreter.module_manager = self.module_manager
            interpreter.module_cache = self.module_cache  # Compartilhar cache
            interpreter._loading_modules = self._loading_modules  # Compartilhar set
            
            try:
                for statement in program.statements:
                    interpreter.eval_statement(statement)
            except ReturnException:
                pass
            
            # Create module object with exported variables
            module_obj = {
                name: value
                for name, value in module_env.variables.items()
                if not name.startswith('_')
            }
            
            # Adicionar ao cache antes de remover do set
            self.module_cache[normalized_path] = module_obj
            
            return module_obj
            
        finally:
            # Remover do set de módulos sendo carregados
            self._loading_modules.discard(normalized_path)
    
    def load_python_module(self, module_path: str, module_name: str) -> dict:
        """Load .py Python module with caching"""
        # Normalizar o caminho do módulo
        normalized_path = str(Path(module_path).resolve())
        
        # Verificar se o módulo já está no cache
        if normalized_path in self.module_cache:
            return self.module_cache[normalized_path]
        
        import importlib.util
        import types
        
        try:
            # Load Python module dynamically
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise HubbleException(f"Cannot load Python module: {module_name}")
            
            py_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(py_module)
            
            # Extract public attributes
            module_obj = {}
            for attr_name in dir(py_module):
                if not attr_name.startswith('_'):
                    attr = getattr(py_module, attr_name)
                    # Include functions, classes, modules, and simple types
                    if callable(attr) or isinstance(attr, (int, float, str, bool, list, dict, type(None), types.ModuleType)):
                        module_obj[attr_name] = attr
            
            # Adicionar ao cache
            self.module_cache[normalized_path] = module_obj
            
            return module_obj
        
        except Exception as e:
            raise HubbleException(f"Error loading Python module '{module_name}': {e}")
    
    def load_python_package(self, module_name: str) -> dict:
        """Load Python package from system (pip installed packages)"""
        # Verificar se já está no cache
        cache_key = f"__python_package__{module_name}"
        if cache_key in self.module_cache:
            return self.module_cache[cache_key]
        
        import importlib
        import types
        
        try:
            # Try to import the Python package
            py_module = importlib.import_module(module_name)
            
            # Extract public attributes
            module_obj = {}
            for attr_name in dir(py_module):
                if not attr_name.startswith('_'):
                    try:
                        attr = getattr(py_module, attr_name)
                        # Include everything that's accessible
                        module_obj[attr_name] = attr
                    except:
                        # Skip attributes that can't be accessed
                        pass
            
            # Adicionar o próprio módulo para permitir acesso completo
            module_obj['__module__'] = py_module
            
            # Adicionar ao cache
            self.module_cache[cache_key] = module_obj
            
            return module_obj
        
        except ImportError as e:
            raise HubbleException(f"Cannot import Python package '{module_name}': {e}. Make sure it's installed with pip.")
        except Exception as e:
            raise HubbleException(f"Error loading Python package '{module_name}': {e}")
    
    def load_native_module(self, module_path: str, module_name: str, module_type: str) -> dict:
        """Load native module (.exe, .dll, .so, .dylib) with full .exe support"""
        import ctypes
        import platform
        
        try:
            module_obj = {}
            
            if module_type == 'exe':
                # Advanced .exe support - can be used as library or executable
                module_obj['path'] = module_path
                
                # Method 1: Execute as process
                def run_exe(*args):
                    import subprocess
                    cmd = [module_path] + [str(arg) for arg in args]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    return {
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode,
                        'success': result.returncode == 0
                    }
                
                module_obj['run'] = run_exe
                module_obj['execute'] = run_exe
                
                # Method 2: Try to load as DLL/library (Windows only)
                if sys.platform == 'win32':
                    try:
                        # Try to load .exe as a library (some .exe files export functions)
                        lib = ctypes.CDLL(module_path)
                        module_obj['_lib'] = lib
                        
                        # Add helper to call exported functions
                        def call_function(func_name, *args, **kwargs):
                            """Call an exported function from the .exe"""
                            if not hasattr(lib, func_name):
                                raise HubbleException(f"Function '{func_name}' not found in {module_name}")
                            
                            func = getattr(lib, func_name)
                            
                            if 'restype' in kwargs:
                                func.restype = kwargs['restype']
                            if 'argtypes' in kwargs:
                                func.argtypes = kwargs['argtypes']
                            
                            return func(*args)
                        
                        module_obj['call'] = call_function
                        module_obj['get_function'] = lambda name: getattr(lib, name)
                        
                        # Add ctypes for type conversion
                        module_obj['c_int'] = ctypes.c_int
                        module_obj['c_float'] = ctypes.c_float
                        module_obj['c_double'] = ctypes.c_double
                        module_obj['c_char_p'] = ctypes.c_char_p
                        module_obj['c_wchar_p'] = ctypes.c_wchar_p
                        module_obj['c_void_p'] = ctypes.c_void_p
                        module_obj['c_bool'] = ctypes.c_bool
                        module_obj['c_size_t'] = ctypes.c_size_t
                        module_obj['POINTER'] = ctypes.POINTER
                        module_obj['byref'] = ctypes.byref
                        module_obj['Structure'] = ctypes.Structure
                        
                        module_obj['_is_library'] = True
                    except (OSError, Exception):
                        # .exe doesn't export functions, only executable mode available
                        module_obj['_is_library'] = False
                        pass
                
                # Method 3: Execute with pipe communication
                def execute_with_input(stdin_data="", *args):
                    """Execute .exe with stdin input"""
                    import subprocess
                    cmd = [module_path] + [str(arg) for arg in args]
                    result = subprocess.run(
                        cmd, 
                        input=stdin_data, 
                        capture_output=True, 
                        text=True
                    )
                    return {
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode,
                        'success': result.returncode == 0
                    }
                
                module_obj['execute_with_input'] = execute_with_input
                
                # Method 4: Execute async (non-blocking)
                def run_async(*args):
                    """Run .exe asynchronously"""
                    import subprocess
                    cmd = [module_path] + [str(arg) for arg in args]
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    return {
                        'process': process,
                        'pid': process.pid,
                        'wait': lambda: process.wait(),
                        'poll': lambda: process.poll(),
                        'kill': lambda: process.kill(),
                        'terminate': lambda: process.terminate(),
                        'communicate': lambda: process.communicate()
                    }
                
                module_obj['run_async'] = run_async
                module_obj['spawn'] = run_async
                
                # Method 5: Check if .exe is running
                def is_running():
                    """Check if process is running"""
                    import psutil
                    try:
                        exe_name = Path(module_path).name
                        for proc in psutil.process_iter(['name']):
                            if proc.info['name'] == exe_name:
                                return True
                        return False
                    except:
                        return False
                
                try:
                    import psutil
                    module_obj['is_running'] = is_running
                except ImportError:
                    pass
                
            elif module_type in ['dll', 'so', 'dylib', 'pyd']:
                # Load shared library
                try:
                    if module_type == 'dll' or module_type == 'pyd':
                        lib = ctypes.CDLL(module_path)
                    elif module_type == 'so':
                        lib = ctypes.CDLL(module_path)
                    elif module_type == 'dylib':
                        lib = ctypes.CDLL(module_path)
                    
                    module_obj['_lib'] = lib
                    module_obj['path'] = module_path
                    
                    # Add helper function to call library functions
                    def call_function(func_name, *args, **kwargs):
                        """Call a function from the native library"""
                        if not hasattr(lib, func_name):
                            raise HubbleException(f"Function '{func_name}' not found in library")
                        
                        func = getattr(lib, func_name)
                        
                        if 'restype' in kwargs:
                            func.restype = kwargs['restype']
                        if 'argtypes' in kwargs:
                            func.argtypes = kwargs['argtypes']
                        
                        return func(*args)
                    
                    module_obj['call'] = call_function
                    module_obj['get_function'] = lambda name: getattr(lib, name)
                    
                    # Add ctypes types for convenience
                    module_obj['c_int'] = ctypes.c_int
                    module_obj['c_uint'] = ctypes.c_uint
                    module_obj['c_long'] = ctypes.c_long
                    module_obj['c_ulong'] = ctypes.c_ulong
                    module_obj['c_longlong'] = ctypes.c_longlong
                    module_obj['c_ulonglong'] = ctypes.c_ulonglong
                    module_obj['c_float'] = ctypes.c_float
                    module_obj['c_double'] = ctypes.c_double
                    module_obj['c_char'] = ctypes.c_char
                    module_obj['c_char_p'] = ctypes.c_char_p
                    module_obj['c_wchar'] = ctypes.c_wchar
                    module_obj['c_wchar_p'] = ctypes.c_wchar_p
                    module_obj['c_void_p'] = ctypes.c_void_p
                    module_obj['c_bool'] = ctypes.c_bool
                    module_obj['c_byte'] = ctypes.c_byte
                    module_obj['c_ubyte'] = ctypes.c_ubyte
                    module_obj['c_short'] = ctypes.c_short
                    module_obj['c_ushort'] = ctypes.c_ushort
                    module_obj['c_size_t'] = ctypes.c_size_t
                    module_obj['c_ssize_t'] = ctypes.c_ssize_t
                    module_obj['POINTER'] = ctypes.POINTER
                    module_obj['pointer'] = ctypes.pointer
                    module_obj['byref'] = ctypes.byref
                    module_obj['Structure'] = ctypes.Structure
                    module_obj['Union'] = ctypes.Union
                    module_obj['Array'] = ctypes.Array
                    module_obj['sizeof'] = ctypes.sizeof
                    module_obj['cast'] = ctypes.cast
                    module_obj['string_at'] = ctypes.string_at
                    module_obj['wstring_at'] = ctypes.wstring_at
                    
                except OSError as e:
                    raise HubbleException(f"Cannot load native library '{module_path}': {e}")
            
            return module_obj
        
        except Exception as e:
            raise HubbleException(f"Error loading native module '{module_name}': {e}")
        """Load any generic file as a module - universal loader"""
        # Normalizar o caminho do módulo
        normalized_path = str(Path(module_path).resolve())
        
        # Verificar se o módulo já está no cache
        if normalized_path in self.module_cache:
            return self.module_cache[normalized_path]
        
        try:
            module_obj = {}
            
            # For script files (bat, sh, cmd, etc.), create execution wrapper
            if file_type in ['bat', 'cmd', 'sh', 'exe']:
                def run_script(*args):
                    import subprocess
                    cmd = [module_path] + [str(arg) for arg in args]
                    result = subprocess.run(cmd, capture_output=True, text=True, shell=(file_type in ['bat', 'cmd']))
                    return {
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode,
                        'success': result.returncode == 0
                    }
                
                module_obj['run'] = run_script
                module_obj['execute'] = run_script
                module_obj['path'] = module_path
            
            # For text-based script files that might be other languages
            elif file_type in ['js', 'mjs', 'lua', 'rb', 'pl']:
                # Create an executor for the specific interpreter
                interpreters = {
                    'js': ['node'],
                    'mjs': ['node'],
                    'lua': ['lua'],
                    'rb': ['ruby'],
                    'pl': ['perl']
                }
                
                interpreter = interpreters.get(file_type, [])
                
                def run_with_interpreter(*args):
                    import subprocess
                    cmd = interpreter + [module_path] + [str(arg) for arg in args]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    return {
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode,
                        'success': result.returncode == 0
                    }
                
                module_obj['run'] = run_with_interpreter
                module_obj['execute'] = run_with_interpreter
                module_obj['path'] = module_path
                module_obj['interpreter'] = interpreter
            
            # For Java files
            elif file_type in ['jar', 'class']:
                def run_java(*args):
                    import subprocess
                    if file_type == 'jar':
                        cmd = ['java', '-jar', module_path] + [str(arg) for arg in args]
                    else:
                        # For .class files, need to run java with classpath
                        cmd = ['java', '-cp', str(Path(module_path).parent), Path(module_path).stem] + [str(arg) for arg in args]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    return {
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode,
                        'success': result.returncode == 0
                    }
                
                module_obj['run'] = run_java
                module_obj['execute'] = run_java
                module_obj['path'] = module_path
            
            # For Python packages (zip, egg, whl)
            elif file_type in ['zip', 'egg', 'whl']:
                # Try to add to Python path and import
                import sys
                if module_path not in sys.path:
                    sys.path.insert(0, module_path)
                
                try:
                    import importlib
                    # Try to import the package
                    pkg = importlib.import_module(Path(module_path).stem)
                    for attr_name in dir(pkg):
                        if not attr_name.startswith('_'):
                            try:
                                module_obj[attr_name] = getattr(pkg, attr_name)
                            except:
                                pass
                except:
                    # If can't import, just provide path access
                    module_obj['path'] = module_path
                    module_obj['__package__'] = True
            
            # For compiled Python files
            elif file_type in ['pyc', 'pyo']:
                import importlib.util
                try:
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec and spec.loader:
                        py_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(py_module)
                        
                        for attr_name in dir(py_module):
                            if not attr_name.startswith('_'):
                                try:
                                    module_obj[attr_name] = getattr(py_module, attr_name)
                                except:
                                    pass
                except:
                    module_obj['path'] = module_path
            
            # For unknown file types, try to read as text or provide raw access
            else:
                try:
                    # Try to read as text
                    with open(module_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    module_obj['content'] = content
                    module_obj['read'] = lambda: content
                except:
                    # Binary file, provide raw read function
                    def read_binary():
                        with open(module_path, 'rb') as f:
                            return f.read()
                    module_obj['read'] = read_binary
                
                module_obj['path'] = module_path
                
                # Try to execute as script if possible
                def try_execute(*args):
                    import subprocess
                    try:
                        cmd = [module_path] + [str(arg) for arg in args]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        return {
                            'stdout': result.stdout,
                            'stderr': result.stderr,
                            'returncode': result.returncode,
                            'success': result.returncode == 0
                        }
                    except Exception as e:
                        return {
                            'error': str(e),
                            'success': False
                        }
                
                module_obj['execute'] = try_execute
            
            # Always provide basic file info
            module_obj['__file__'] = module_path
            module_obj['__type__'] = file_type
            module_obj['__name__'] = module_name
            
            # Cache the module
            self.module_cache[normalized_path] = module_obj
            
            return module_obj
        
        except Exception as e:
            raise HubbleException(f"Error loading module '{module_name}' from '{module_path}': {e}")
        """Load native module (.exe, .dll, .so, .dylib)"""
        import ctypes
        import platform
        
        try:
            module_obj = {}
            
            if module_type == 'exe':
                # For .exe files, create a wrapper that executes the program
                def run_exe(*args):
                    import subprocess
                    cmd = [module_path] + [str(arg) for arg in args]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    return {
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode,
                        'success': result.returncode == 0
                    }
                
                module_obj['run'] = run_exe
                module_obj['path'] = module_path
                module_obj['execute'] = run_exe
                
            elif module_type in ['dll', 'so', 'dylib']:
                # Load shared library
                try:
                    if module_type == 'dll':
                        lib = ctypes.CDLL(module_path)
                    elif module_type == 'so':
                        lib = ctypes.CDLL(module_path)
                    elif module_type == 'dylib':
                        lib = ctypes.CDLL(module_path)
                    
                    # Create wrapper object
                    module_obj['_lib'] = lib
                    module_obj['path'] = module_path
                    
                    # Add helper function to call library functions
                    def call_function(func_name, *args, **kwargs):
                        """Call a function from the native library"""
                        if not hasattr(lib, func_name):
                            raise HubbleException(f"Function '{func_name}' not found in library")
                        
                        func = getattr(lib, func_name)
                        
                        # Setup return type if specified
                        if 'restype' in kwargs:
                            func.restype = kwargs['restype']
                        
                        # Setup argument types if specified
                        if 'argtypes' in kwargs:
                            func.argtypes = kwargs['argtypes']
                        
                        return func(*args)
                    
                    module_obj['call'] = call_function
                    module_obj['get_function'] = lambda name: getattr(lib, name)
                    
                    # Add ctypes types for convenience
                    module_obj['c_int'] = ctypes.c_int
                    module_obj['c_float'] = ctypes.c_float
                    module_obj['c_double'] = ctypes.c_double
                    module_obj['c_char_p'] = ctypes.c_char_p
                    module_obj['c_void_p'] = ctypes.c_void_p
                    module_obj['c_bool'] = ctypes.c_bool
                    module_obj['POINTER'] = ctypes.POINTER
                    
                except OSError as e:
                    raise HubbleException(f"Cannot load native library '{module_path}': {e}")
            
            return module_obj
        
        except Exception as e:
            raise HubbleException(f"Error loading native module '{module_name}': {e}")
    
    def find_module_with_type(self, name: str) -> Tuple[Optional[str], Optional[str]]:
        """Find module file with support for .hbl, .py, .exe, .dll, .so, .dylib - ANY FILE TYPE"""
        # Support for package.submodule syntax
        parts = name.split('.')
        base_name = name.replace(".", "/")
        
        # Define ALL possible extensions - not limited!
        # Standard extensions
        extensions = ['.hbl', '.py']
        
        # Platform-specific binary extensions
        if sys.platform == 'win32':
            extensions.extend(['.exe', '.dll', '.pyd', '.pyw'])
        elif sys.platform == 'darwin':
            extensions.extend(['.dylib', '.so'])
        else:  # Linux and other Unix-like
            extensions.extend(['.so'])
        
        # Additional extensions that might be modules
        # Python compiled, libraries, scripts, etc.
        extensions.extend([
            '.pyc', '.pyo', '.pyi',  # Python bytecode and stubs
            '.zip', '.egg', '.whl',   # Python packages
            '.js', '.mjs',            # JavaScript (if needed)
            '.lua',                   # Lua scripts
            '.rb',                    # Ruby scripts
            '.pl',                    # Perl scripts
            '.sh', '.bat', '.cmd',    # Shell scripts
            '.jar',                   # Java archives
            '.class',                 # Java classes
        ])
        
        search_paths = []
        
        # 1. Check current directory and subdirectories (HIGHEST PRIORITY)
        search_paths.append(Path.cwd())
        
        # 2. Check standard library
        stdlib_dir = Path(__file__).parent / 'stdlib'
        if stdlib_dir.exists():
            search_paths.append(stdlib_dir)
        
        # 3. Check HUBBLE_PATH environment variable
        hubble_path = os.environ.get('HUBBLE_PATH', '')
        if hubble_path:
            for path in hubble_path.split(os.pathsep):
                search_paths.append(Path(path))
        
        # Search for module file with known extensions
        for search_path in search_paths:
            for ext in extensions:
                # Direct file match
                module_file = search_path / f"{base_name}{ext}"
                if module_file.exists():
                    return str(module_file), ext[1:]  # Return path and extension without dot
                
                # Simple name match
                simple_file = search_path / f"{name}{ext}"
                if simple_file.exists():
                    return str(simple_file), ext[1:]
        
        # NEW: Search for ANY file with the exact name (no extension requirement)
        # This allows importing files like "mylibrary" without extension
        for search_path in search_paths:
            # Try exact name match
            exact_file = search_path / name
            if exact_file.exists() and exact_file.is_file():
                # Detect type by trying to read it
                file_type = self.detect_file_type(str(exact_file))
                if file_type:
                    return str(exact_file), file_type
            
            # Try with path separators
            path_file = search_path / base_name
            if path_file.exists() and path_file.is_file():
                file_type = self.detect_file_type(str(path_file))
                if file_type:
                    return str(path_file), file_type
        
        # NEW: Wildcard search - find ANY file starting with the module name
        for search_path in search_paths:
            if search_path.exists():
                try:
                    # Search for files matching the pattern
                    for file_path in search_path.glob(f"{name}*"):
                        if file_path.is_file():
                            # Check if it's a valid module file
                            ext = file_path.suffix
                            if ext:
                                return str(file_path), ext[1:]
                            else:
                                # File without extension, try to detect type
                                file_type = self.detect_file_type(str(file_path))
                                if file_type:
                                    return str(file_path), file_type
                except:
                    pass
        
        # Check for package __init__ file
        if len(parts) > 1:
            for search_path in search_paths:
                package_init = search_path / parts[0] / '__init__.hbl'
                if package_init.exists():
                    return str(package_init), 'hbl'
                
                # Also check for Python __init__.py
                package_init_py = search_path / parts[0] / '__init__.py'
                if package_init_py.exists():
                    return str(package_init_py), 'py'
        
        return None, None
    
    def detect_file_type(self, file_path: str) -> Optional[str]:
        """Detect file type by reading its content or checking attributes"""
        try:
            # Check if it's executable
            if os.access(file_path, os.X_OK):
                if sys.platform == 'win32':
                    return 'exe'
                else:
                    # Unix executable, treat as shell script or binary
                    return 'exe'
            
            # Try to detect by reading first few bytes
            with open(file_path, 'rb') as f:
                header = f.read(100)
                
                # Check for Python file
                if header.startswith(b'#!') and b'python' in header:
                    return 'py'
                
                # Check for shell script
                if header.startswith(b'#!/bin/sh') or header.startswith(b'#!/bin/bash'):
                    return 'exe'
                
                # Check for text file that might be Python
                try:
                    header_text = header.decode('utf-8')
                    if 'import ' in header_text or 'def ' in header_text or 'class ' in header_text:
                        return 'py'
                    # Check for Hubble syntax
                    if 'func ' in header_text or 'var ' in header_text or 'end' in header_text:
                        return 'hbl'
                except:
                    pass
                
                # Check for binary library signatures
                if header.startswith(b'MZ'):  # Windows PE
                    return 'dll'
                if header.startswith(b'\x7fELF'):  # Linux ELF
                    return 'so'
                if b'Mach-O' in header or header.startswith(b'\xca\xfe\xba\xbe'):  # macOS Mach-O
                    return 'dylib'
            
            # Default: try to treat as Python module
            return 'py'
        except:
            return None
    
    def find_module(self, name: str) -> Optional[str]:
        """Find module file (legacy method, kept for compatibility)"""
        module_path, _ = self.find_module_with_type(name)
        return module_path
    
    def eval_try_statement(self, node: TryStatement):
        """Evaluate try-catch-finally"""
        exception_caught = False
        exception_value = None
        
        # Try block
        try:
            for stmt in node.try_block:
                self.eval_statement(stmt)
        except HubbleException as e:
            exception_caught = True
            exception_value = str(e)
        except Exception as e:
            exception_caught = True
            exception_value = str(e)
        
        # Catch block
        if exception_caught and node.catch_block:
            catch_env = Environment(parent=self.current_env)
            if node.catch_var:
                catch_env.define(node.catch_var, exception_value)
            
            prev_env = self.current_env
            self.current_env = catch_env
            
            try:
                for stmt in node.catch_block:
                    self.eval_statement(stmt)
            finally:
                self.current_env = prev_env
        
        # Finally block
        if node.finally_block:
            for stmt in node.finally_block:
                self.eval_statement(stmt)
        
        # Re-raise if not caught
        if exception_caught and not node.catch_block:
            raise HubbleException(exception_value)
    
    def eval_throw(self, node: ThrowStatement):
        """Evaluate throw statement"""
        value = self.eval_expression(node.value)
        raise HubbleException(str(value))
    
    def eval_match_statement(self, node: MatchStatement):
        """Evaluate match statement"""
        value = self.eval_expression(node.value)
        
        # Try each case
        for pattern, body in node.cases:
            pattern_value = self.eval_expression(pattern)
            if value == pattern_value:
                for stmt in body:
                    self.eval_statement(stmt)
                return
        
        # Default case
        for stmt in node.default_case:
            self.eval_statement(stmt)
    
    def eval_with_statement(self, node: WithStatement):
        """Evaluate with statement"""
        context = self.eval_expression(node.context)
        
        with_env = Environment(parent=self.current_env)
        if node.variable:
            with_env.define(node.variable, context)
        
        prev_env = self.current_env
        self.current_env = with_env
        
        try:
            # Call __enter__ if exists
            if isinstance(context, HubbleInstance) and 'enter' in context.klass.methods:
                enter_method = context.klass.methods['enter']
                self.call_function(enter_method, [], context)
            
            # Execute body
            for stmt in node.body:
                self.eval_statement(stmt)
        finally:
            # Call __exit__ if exists
            if isinstance(context, HubbleInstance) and 'exit' in context.klass.methods:
                exit_method = context.klass.methods['exit']
                self.call_function(exit_method, [], context)
            
            self.current_env = prev_env


# ============================================================================
# MODULE SYSTEM AND STANDARD LIBRARY
# ============================================================================

class ModuleManager:
    """Manages Hubble modules and standard library"""
    
    def __init__(self):
        self.stdlib_modules = {
            'math': self.create_math_module(),
            'string': self.create_string_module(),
            'array': self.create_array_module(),
            'file': self.create_file_module(),
            'json': self.create_json_module(),
            'random': self.create_random_module(),
            'time': self.create_time_module(),
            'sys': self.create_sys_module(),
        }
    
    def create_math_module(self) -> dict:
        """Create math module"""
        return {
            'PI': math.pi,
            'E': math.e,
            'TAU': math.tau,
            'INF': math.inf,
            'NAN': math.nan,
            'abs': abs,
            'sqrt': math.sqrt,
            'pow': math.pow,
            'floor': math.floor,
            'ceil': math.ceil,
            'round': round,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'atan2': math.atan2,
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'exp': math.exp,
            'degrees': math.degrees,
            'radians': math.radians,
            'factorial': math.factorial,
            'gcd': math.gcd,
            'lcm': lambda a, b: abs(a * b) // math.gcd(a, b) if a and b else 0,
        }
    
    def create_string_module(self) -> dict:
        """Create string module"""
        return {
            'upper': str.upper,
            'lower': str.lower,
            'title': str.title,
            'capitalize': str.capitalize,
            'strip': str.strip,
            'lstrip': str.lstrip,
            'rstrip': str.rstrip,
            'split': str.split,
            'join': lambda sep, arr: sep.join(str(x) for x in arr),
            'replace': str.replace,
            'startswith': str.startswith,
            'endswith': str.endswith,
            'find': str.find,
            'index': str.index,
            'count': str.count,
            'isdigit': str.isdigit,
            'isalpha': str.isalpha,
            'isalnum': str.isalnum,
            'isspace': str.isspace,
            'isupper': str.isupper,
            'islower': str.islower,
        }
    
    def create_array_module(self) -> dict:
        """Create array module"""
        return {
            'push': lambda arr, item: arr.append(item) or arr,
            'pop': list.pop,
            'shift': lambda arr: arr.pop(0) if arr else None,
            'unshift': lambda arr, item: arr.insert(0, item) or arr,
            'slice': lambda arr, start, end=None: arr[start:end],
            'reverse': lambda arr: list(reversed(arr)),
            'sort': sorted,
            'filter': lambda func, arr: list(filter(func, arr)),
            'map': lambda func, arr: list(map(func, arr)),
            'reduce': lambda func, arr, init=None: __import__('functools').reduce(func, arr, init) if init is not None else __import__('functools').reduce(func, arr),
            'find': lambda func, arr: next((x for x in arr if func(x)), None),
            'find_index': lambda func, arr: next((i for i, x in enumerate(arr) if func(x)), -1),
            'every': lambda func, arr: all(func(x) for x in arr),
            'some': lambda func, arr: any(func(x) for x in arr),
            'flatten': lambda arr: [item for sublist in arr for item in (sublist if isinstance(sublist, list) else [sublist])],
            'unique': lambda arr: list(dict.fromkeys(arr)),
        }
    
    def create_file_module(self) -> dict:
        """Create file module"""
        return {
            'read': lambda path: open(path, 'r', encoding='utf-8').read(),
            'write': lambda path, content: open(path, 'w', encoding='utf-8').write(content),
            'append': lambda path, content: open(path, 'a', encoding='utf-8').write(content),
            'exists': os.path.exists,
            'is_file': os.path.isfile,
            'is_dir': os.path.isdir,
            'list_dir': os.listdir,
            'mkdir': os.makedirs,
            'remove': os.remove,
            'rename': os.rename,
            'get_size': os.path.getsize,
        }
    
    def create_json_module(self) -> dict:
        """Create JSON module"""
        return {
            'parse': json.loads,
            'stringify': lambda obj, indent=None: json.dumps(obj, indent=indent),
            'load': lambda path: json.load(open(path, 'r', encoding='utf-8')),
            'save': lambda path, obj, indent=None: json.dump(obj, open(path, 'w', encoding='utf-8'), indent=indent),
        }
    
    def create_random_module(self) -> dict:
        """Create random module"""
        return {
            'random': random.random,
            'randint': random.randint,
            'uniform': random.uniform,
            'choice': random.choice,
            'choices': random.choices,
            'shuffle': lambda arr: random.shuffle(arr) or arr,
            'sample': random.sample,
            'seed': random.seed,
        }
    
    def create_time_module(self) -> dict:
        """Create time module"""
        return {
            'time': time.time,
            'sleep': time.sleep,
            'perf_counter': time.perf_counter,
            'strftime': time.strftime,
            'strptime': lambda s, fmt: time.strptime(s, fmt),
            'localtime': time.localtime,
            'gmtime': time.gmtime,
        }
    
    def create_sys_module(self) -> dict:
        """Create sys module"""
        return {
            'exit': sys.exit,
            'argv': sys.argv,
            'version': '1.0.0',
            'platform': sys.platform,
            'path': sys.path,
        }


# ============================================================================
# CLI AND MAIN ENTRY POINT
# ============================================================================

class HubbleCLI:
    """Command-line interface for Hubble"""
    
    VERSION = "1.0.0"
    
    def __init__(self):
        self.interpreter = Interpreter()
        self.module_manager = ModuleManager()
    
    def run_file(self, filepath: str):
        """Run Hubble file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            
            self.run_source(source, filepath)
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
    
    def run_source(self, source: str, filename: str = '<stdin>'):
        """Run Hubble source code"""
        try:
            # Lexical analysis
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            
            # Parsing
            parser = Parser(tokens)
            program = parser.parse()
            
            # Interpretation
            self.interpreter.interpret(program)
        except SyntaxError as e:
            print(f"Syntax Error in {filename}: {e}", file=sys.stderr)
            sys.exit(1)
        except HubbleException as e:
            print(f"Runtime Error in {filename}: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Internal Error in {filename}: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
    
    def repl(self):
        """Interactive REPL"""
        print(f"Hubble v{self.VERSION}")
        print("Type 'exit()' or press Ctrl+C to quit")
        print()
        
        while True:
            try:
                source = input(">>> ")
                if not source.strip():
                    continue
                
                if source.strip() == 'exit()':
                    break
                
                # Try to evaluate as expression first
                try:
                    lexer = Lexer(source)
                    tokens = lexer.tokenize()
                    parser = Parser(tokens)
                    
                    # Parse as expression
                    parser.skip_newlines()
                    if parser.current_token.type != TokenType.EOF:
                        expr = parser.parse_expression()
                        result = self.interpreter.eval_expression(expr)
                        if result is not None:
                            print(result)
                except:
                    # If expression fails, try as statement
                    self.run_source(source, '<repl>')
            
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                break
            except EOFError:
                print()
                break
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
    
    def show_help(self):
        """Show help message"""
        help_text = f"""
Hubble Programming Language v{self.VERSION}

Usage:
    hbl -- "<file.hbl>"         Run a Hubble file
    hbl --repl                   Start interactive REPL
    hbl --version                Show version
    hbl --help                   Show this help message

Examples:
    hbl -- "hello.hbl"           Run hello.hbl
    hbl -- "examples/demo.hbl"   Run demo.hbl from examples folder

Language Features:
    - High-level syntax similar to Python and Lua
    - Dynamic typing
    - Functions and lambdas
    - Classes and inheritance
    - Exception handling (try/catch/finally)
    - Pattern matching
    - Module system
    - Rich standard library

Documentation: https://hubble-lang.org/docs
"""
        print(help_text)
    
    def show_version(self):
        """Show version"""
        print(f"Hubble v{self.VERSION}")


def setup_system():
    """Setup Hubble in system PATH"""
    print("Setting up Hubble Programming Language...")
    
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    script_path = script_dir / "hubble.py"
    
    # Create hbl command wrapper
    if sys.platform == "win32":
        # Windows batch file
        batch_content = f"""@echo off
python "{script_path}" %*
"""
        batch_path = script_dir / "hbl.bat"
        with open(batch_path, 'w') as f:
            f.write(batch_content)
        
        # Add to PATH
        import winreg
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r'Environment',
                0,
                winreg.KEY_ALL_ACCESS
            )
            
            current_path = winreg.QueryValueEx(key, 'PATH')[0]
            if str(script_dir) not in current_path:
                new_path = f"{current_path};{script_dir}"
                winreg.SetValueEx(key, 'PATH', 0, winreg.REG_EXPAND_SZ, new_path)
                print(f"✓ Added {script_dir} to PATH")
            else:
                print(f"✓ {script_dir} already in PATH")
            
            winreg.CloseKey(key)
            print("\n✓ Setup complete!")
            print("Please restart your command prompt for changes to take effect.")
            print("\nUsage: hbl -- \"<file.hbl>\"")
        except Exception as e:
            print(f"Error setting up PATH: {e}")
            print(f"\nPlease manually add {script_dir} to your PATH")
    else:
        # Unix-like systems (Linux, macOS)
        shell_script = f"""#!/bin/bash
python3 "{script_path}" "$@"
"""
        shell_path = script_dir / "hbl"
        with open(shell_path, 'w') as f:
            f.write(shell_script)
        
        # Make executable
        os.chmod(shell_path, 0o755)
        
        # Add to PATH in shell config
        home = Path.home()
        shell_configs = [
            home / ".bashrc",
            home / ".bash_profile",
            home / ".zshrc",
        ]
        
        path_line = f'\nexport PATH="$PATH:{script_dir}"\n'
        
        for config in shell_configs:
            if config.exists():
                with open(config, 'r') as f:
                    content = f.read()
                
                if str(script_dir) not in content:
                    with open(config, 'a') as f:
                        f.write(path_line)
                    print(f"✓ Added to {config}")
        
        print("\n✓ Setup complete!")
        print("Please run: source ~/.bashrc (or your shell config)")
        print("\nUsage: hbl -- \"<file.hbl>\"")


def main():
    """Main entry point"""
    cli = HubbleCLI()
    
    # If run directly without arguments, setup
    if len(sys.argv) == 1:
        setup_system()
        return
    
    # Parse command-line arguments
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        
        if arg == "--help" or arg == "-h":
            cli.show_help()
        elif arg == "--version" or arg == "-v":
            cli.show_version()
        elif arg == "--repl" or arg == "-r":
            cli.repl()
        elif arg == "--":
            # Run file: hbl -- "file.hbl"
            if len(sys.argv) >= 3:
                filepath = sys.argv[2]
                cli.run_file(filepath)
            else:
                print("Error: No file specified", file=sys.stderr)
                print("Usage: hbl -- \"<file.hbl>\"", file=sys.stderr)
                sys.exit(1)
        else:
            # Assume it's a file
            cli.run_file(arg)
    else:
        cli.show_help()


if __name__ == "__main__":
    main()
