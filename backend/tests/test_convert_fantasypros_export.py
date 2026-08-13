"""
Tests for convert_fantasypros_export.py's row-parsing logic — the raw
FantasyPros "Player (Bye)" column is the risky part (name/team/bye packed
into one field, defenses formatted differently again), so this pins the
parsing rather than the file I/O around it.
"""

import csv
import io

from backend.ingestion.convert_fantasypros_export import (
    convert_rows,
    parse_player_bye,
)


# ---------------------------------------------------------------------------
# parse_player_bye
# ---------------------------------------------------------------------------

def test_parses_a_normal_player_row():
    assert parse_player_bye("Jahmyr Gibbs   DET (6)") == ("Jahmyr Gibbs", "DET", 6)


def test_parses_names_with_suffixes_and_apostrophes():
    assert parse_player_bye("James Cook III   BUF (7)") == ("James Cook III", "BUF", 7)
    assert parse_player_bye("Ja'Marr Chase   CIN (6)") == ("Ja'Marr Chase", "CIN", 6)
    assert parse_player_bye("Travis Etienne Jr.   NO (8)") == ("Travis Etienne Jr.", "NO", 8)


def test_normalises_team_codes_the_export_gets_wrong():
    # FantasyPros labels Jacksonville "JAC"; Player.team elsewhere (and the
    # DST rows in this same export) use "JAX". Confirmed live on the 2026
    # export — 14 Jaguars rows would otherwise silently fail every
    # team-keyed join downstream (Game, roster changes, OPP tags).
    name, team, bye = parse_player_bye("Trevor Lawrence   JAC (7)")
    assert team == "JAX"


def test_parses_a_defense_row_and_maps_the_full_name_to_a_code():
    name, team, bye = parse_player_bye("Houston Texans DST   (8)")
    assert team == "HOU"
    assert bye == 8
    assert "DST" in name


def test_defense_row_handles_a_multi_word_city():
    name, team, bye = parse_player_bye("Los Angeles Rams DST   (11)")
    assert team == "LAR"
    assert bye == 11


def test_defense_row_handles_a_team_name_with_a_number_in_it():
    # "San Francisco 49ers" — the digits must not get mistaken for the bye.
    name, team, bye = parse_player_bye("San Francisco 49ers DST   (8)")
    assert team == "SF"
    assert bye == 8


def test_falls_back_to_bare_name_when_no_team_or_bye_present():
    # A handful of inactive/free-agent veterans near the bottom of a real
    # export carry no team at all (confirmed: Kareem Hunt, Justin Tucker,
    # Philip Rivers on the 2026 sheet). Kept, not dropped — rank/POS/AVG
    # are still real data.
    assert parse_player_bye("Tyreek Hill") == ("Tyreek Hill", "", None)


# ---------------------------------------------------------------------------
# convert_rows
# ---------------------------------------------------------------------------

def _reader(text: str) -> csv.DictReader:
    return csv.DictReader(io.StringIO(text))


def test_convert_rows_produces_the_ingest_players_shape():
    raw = (
        "Rank,Player (Bye),POS,ESPN,Sleeper,CBS,NFL,RTSports,Fantrax,AVG,Real-Time\n"
        "1,Jahmyr Gibbs   DET (6),RB1,1,1,1,,1,1,1.0,1\n"
    )
    rows = convert_rows(_reader(raw))
    assert rows == [{
        "Rank": "1", "Player": "Jahmyr Gibbs", "Team": "DET", "Bye": 6,
        "POS": "RB1", "AVG": "1.0",
    }]


def test_convert_rows_skips_rows_missing_required_fields():
    raw = (
        "Rank,Player (Bye),POS,AVG\n"
        "1,Jahmyr Gibbs   DET (6),RB1,1.0\n"
        ",Missing Rank   DET (6),RB2,2.0\n"
        "3,,RB3,3.0\n"
        "4,No Pos   DET (6),,4.0\n"
        "5,No Avg   DET (6),RB5,\n"
    )
    rows = convert_rows(_reader(raw))
    assert len(rows) == 1
    assert rows[0]["Player"] == "Jahmyr Gibbs"


def test_convert_rows_keeps_rows_with_no_team_rather_than_dropping_them():
    raw = (
        "Rank,Player (Bye),POS,AVG\n"
        "187,Tyreek Hill,WR66,206.4\n"
    )
    rows = convert_rows(_reader(raw))
    assert len(rows) == 1
    assert rows[0]["Team"] == ""
    assert rows[0]["Bye"] == ""


def test_convert_rows_real_time_column_extra_text_does_not_shift_columns():
    # The export sometimes packs a rank-movement delta into the same cell
    # ("8  -1") with no comma — a real value in a column this converter
    # never reads, not a CSV-parsing hazard.
    raw = (
        "Rank,Player (Bye),POS,ESPN,Sleeper,CBS,NFL,RTSports,Fantrax,AVG,Real-Time\n"
        "7,Jonathan Taylor   IND (13),RB4,7,7,5,,7,7,6.6,8  -1\n"
    )
    rows = convert_rows(_reader(raw))
    assert rows[0]["Player"] == "Jonathan Taylor"
    assert rows[0]["Team"] == "IND"
    assert rows[0]["Bye"] == 13
    assert rows[0]["AVG"] == "6.6"
