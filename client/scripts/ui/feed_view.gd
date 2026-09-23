extends RichTextLabel
## The live event feed: one summary line per event with its sim time, newest at the bottom,
## capped at MAX_LINES. Entries are kept sorted by event seq so REST backfill merges cleanly.

const UiTheme := preload("res://scripts/ui/ui_theme.gd")

const MAX_LINES := 200
## Too chatty for the feed (every room change).
const SKIP_TYPES := ["employee.entered"]
const TYPE_COLORS := {
	"speech.said": "2a1f33", "employee.introduced": "2a1f33",
	"owner.message": "c2417a", "owner.reply": "c2417a", "owner.gift": "c2417a",
	"question.submitted": "2f6f9f", "question.claimed": "2f6f9f", "question.draft_updated": "2f6f9f",
	"question.answered": "2f6f9f",
	"employee.arrived": "3f7f52", "employee.left": "3f7f52", "employee.hired": "3f7f52",
	"employee.identity_chosen": "3f7f52", "day.started": "8a5a3a", "day.ended": "8a5a3a",
	"office.lunch": "8a5a3a", "office.workday_end": "8a5a3a",
}

var _entries: Array = []      # [{seq, line}] sorted by seq
var _seqs: Dictionary = {}


func _init() -> void:
	bbcode_enabled = true
	scroll_following = true
	selection_enabled = true
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	autowrap_mode = TextServer.AUTOWRAP_WORD_SMART


## Adds an event; `time_text` is its sim time ("9:05 AM").
func add_event(ev: Dictionary, time_text: String) -> void:
	var type: String = str(ev.get("type", ""))
	var summary: String = str(ev.get("summary", ""))
	var seq: int = int(ev.get("seq", 0))
	if summary == "" or type in SKIP_TYPES or _seqs.has(seq):
		return
	_seqs[seq] = true
	var color: String = TYPE_COLORS.get(type, "5b4a6b")
	if str(ev.get("visibility", "")) == "system":
		color = "8a7f96"
	var line: String = "[color=#9a8fa6]%s[/color]  [color=#%s]%s[/color]" % [
		time_text.trim_suffix(" AM").trim_suffix(" PM"), color, _escape(summary)]
	var entry: Dictionary = {"seq": seq, "line": line}
	if _entries.is_empty() or seq > int(_entries[-1]["seq"]):
		_entries.append(entry)
		if _entries.size() > MAX_LINES + 25:
			_trim_and_render()
		else:
			append_text(("\n" if _entries.size() > 1 else "") + line)
		return
	var at: int = _entries.bsearch_custom(seq, func(e: Dictionary, s: int) -> bool: return int(e["seq"]) < s)
	_entries.insert(at, entry)
	_trim_and_render()


func reset() -> void:
	_entries.clear()
	_seqs.clear()
	clear()


func _trim_and_render() -> void:
	while _entries.size() > MAX_LINES:
		_seqs.erase(int(_entries[0]["seq"]))
		_entries.remove_at(0)
	var lines: PackedStringArray = PackedStringArray()
	for e: Dictionary in _entries:
		lines.append(str(e["line"]))
	clear()
	append_text("\n".join(lines))


static func _escape(text: String) -> String:
	return text.replace("[", "[lb]")
