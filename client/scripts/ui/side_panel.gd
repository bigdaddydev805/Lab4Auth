extends PanelContainer
## The collapsible right-side panel: the inspector card (when someone is selected) above tabs
## for the live Feed, the Team, Questions and the meeting-room Board (whiteboard).

const InspectorCard := preload("res://scripts/ui/inspector_card.gd")
const FeedView := preload("res://scripts/ui/feed_view.gd")
const TeamList := preload("res://scripts/ui/team_list.gd")
const QuestionsView := preload("res://scripts/ui/questions_view.gd")

signal employee_selected(emp_id: String)
signal inspector_closed

const WIDTH := 290

var inspector: InspectorCard
var feed: FeedView
var team: TeamList
var questions: QuestionsView
var tabs: TabContainer

var _board: Label


func _init() -> void:
	mouse_filter = Control.MOUSE_FILTER_STOP
	var col: VBoxContainer = VBoxContainer.new()
	col.add_theme_constant_override("separation", 8)
	add_child(col)

	inspector = InspectorCard.new()
	inspector.visible = false
	inspector.closed.connect(func() -> void: inspector_closed.emit())
	col.add_child(inspector)

	tabs = TabContainer.new()
	tabs.size_flags_vertical = Control.SIZE_EXPAND_FILL
	tabs.tab_focus_mode = Control.FOCUS_NONE
	col.add_child(tabs)

	feed = FeedView.new()
	feed.name = "Feed"
	tabs.add_child(feed)

	team = TeamList.new()
	team.name = "Team"
	team.employee_selected.connect(func(id: String) -> void: employee_selected.emit(id))
	tabs.add_child(team)

	questions = QuestionsView.new()
	questions.name = "Questions"
	tabs.add_child(questions)

	var board_scroll: ScrollContainer = ScrollContainer.new()
	board_scroll.name = "Board"
	board_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	tabs.add_child(board_scroll)
	_board = Label.new()
	_board.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_board.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_board.custom_minimum_size.x = 100
	board_scroll.add_child(_board)
	set_whiteboard("")


func set_whiteboard(text: String) -> void:
	_board.text = text if text.strip_edges() != "" else "The meeting-room whiteboard is blank."
	_board.theme_type_variation = "Label" if text.strip_edges() != "" else "Muted"


func show_tab(tab_name: String) -> void:
	for i: int in range(tabs.get_tab_count()):
		if tabs.get_tab_title(i) == tab_name:
			tabs.current_tab = i
