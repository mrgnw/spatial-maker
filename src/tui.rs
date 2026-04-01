use std::io::{self, stdout};
use std::time::{Duration, Instant};

use crossterm::{
	execute,
	terminal::{disable_raw_mode, enable_raw_mode},
};
use ratatui::{
	layout::{Constraint, Layout, Rect},
	style::{Color, Style},
	text::{Line, Span},
	widgets::{Gauge, Paragraph, Widget},
	Frame, Terminal, TerminalOptions, Viewport,
};

use crate::VideoProgress;

#[derive(Clone, Debug, PartialEq)]
pub enum FileStatus {
	Pending,
	Processing,
	Done { outputs: Vec<String>, duration: Duration },
	Error(String),
}

#[derive(Clone, Debug)]
pub struct FileState {
	pub filename: String,
	pub status: FileStatus,
	pub media_type: MediaType,
	pub stage: Option<String>,
	pub stage_progress: f64,
	pub started_at: Option<Instant>,
	pub stage_started_at: Option<Instant>,
	pub video_frame: u32,
	pub video_total: u32,
	pub video_fps: f64,
	pub video_eta: String,
}

#[derive(Clone, Debug)]
pub enum MediaType {
	Photo,
	Video,
}

pub struct AppState {
	pub files: Vec<FileState>,
	pub current_index: usize,
	pub model_name: String,
	pub model_mb: u32,
	pub total: usize,
	pub done: usize,
	pub start: Instant,
	pub avg_depth_time: Option<f64>,
	pub depth_count: usize,
}

impl AppState {
	pub fn new(filenames: Vec<(String, MediaType)>, model_name: &str, model_mb: u32) -> Self {
		let total = filenames.len();
		let files = filenames
			.into_iter()
			.map(|(filename, media_type)| FileState {
				filename,
				status: FileStatus::Pending,
				media_type,
				stage: None,
				stage_progress: 0.0,
				started_at: None,
				stage_started_at: None,
				video_frame: 0,
				video_total: 0,
				video_fps: 0.0,
				video_eta: String::new(),
			})
			.collect();

		Self {
			files,
			current_index: 0,
			model_name: model_name.to_string(),
			model_mb,
			total,
			done: 0,
			start: Instant::now(),
			avg_depth_time: None,
			depth_count: 0,
		}
	}

	pub fn mark_processing(&mut self, index: usize) {
		if let Some(f) = self.files.get_mut(index) {
			f.status = FileStatus::Processing;
			f.started_at = Some(Instant::now());
		}
		self.current_index = index;
	}

	pub fn update_stage(&mut self, index: usize, stage: String, progress: f64) {
		if let Some(f) = self.files.get_mut(index) {
			if f.stage.as_ref() != Some(&stage) {
				if f.stage.as_deref() == Some("estimating depth") {
					if let Some(started) = f.stage_started_at {
						let elapsed = started.elapsed().as_secs_f64();
						if elapsed > 0.1 {
							self.depth_count += 1;
							self.avg_depth_time = Some(
								self.avg_depth_time
									.map(|avg| (avg * (self.depth_count - 1) as f64 + elapsed) / self.depth_count as f64)
									.unwrap_or(elapsed)
							);
						}
					}
				}
				f.stage_started_at = Some(Instant::now());
			}
			f.stage = Some(stage);
			f.stage_progress = progress;
		}
	}

	pub fn mark_done(&mut self, index: usize, outputs: Vec<String>, duration: Duration) {
		if let Some(f) = self.files.get_mut(index) {
			f.status = FileStatus::Done { outputs, duration };
		}
		self.done += 1;
	}

	pub fn mark_error(&mut self, index: usize, error: String) {
		if let Some(f) = self.files.get_mut(index) {
			f.status = FileStatus::Error(error);
		}
		self.done += 1;
	}

	pub fn update_video_progress(&mut self, index: usize, progress: &VideoProgress, fps: f64, eta: String) {
		if let Some(f) = self.files.get_mut(index) {
			if f.stage.as_ref() != Some(&progress.stage) {
				f.stage_started_at = Some(Instant::now());
			}
			f.stage = Some(progress.stage.clone());
			f.stage_progress = progress.percent;
			f.video_frame = progress.current_frame;
			f.video_total = progress.total_frames;
			f.video_fps = fps;
			f.video_eta = eta;
		}
	}
}

pub fn draw(frame: &mut Frame, state: &AppState) {
	let area = frame.area();

	let chunks = Layout::vertical([
		Constraint::Length(1), // header
		Constraint::Length(1), // gauge
		Constraint::Min(0),   // file list
	])
	.split(area);

	draw_header(frame, chunks[0], state);
	draw_gauge(frame, chunks[1], state);
	draw_files(frame, chunks[2], state);
}

fn draw_header(frame: &mut Frame, area: Rect, state: &AppState) {
	let header = Line::from(vec![
		Span::styled("spatial-maker", Style::default().fg(Color::Cyan).bold()),
		Span::raw("  depth-anything-v2-"),
		Span::styled(&state.model_name, Style::default().bold()),
		Span::styled(
			format!(" ({} MB)", state.model_mb),
			Style::default().fg(Color::DarkGray),
		),
	]);
	frame.render_widget(Paragraph::new(header), area);
}

fn draw_gauge(frame: &mut Frame, area: Rect, state: &AppState) {
	let current_file_progress = if state.current_index < state.files.len() {
		let f = &state.files[state.current_index];
		if f.status == FileStatus::Processing {
			stage_progress_weight(f.stage.as_deref())
		} else {
			0.0
		}
	} else {
		0.0
	};

	let ratio = if state.total > 0 {
		(state.done as f64 + current_file_progress) / state.total as f64
	} else {
		0.0
	};
	
	let label = format!("{}/{}", state.done, state.total);
	let gauge = Gauge::default()
		.gauge_style(Style::default().fg(Color::Cyan).bg(Color::DarkGray))
		.ratio(ratio.min(1.0))
		.label(label);
	frame.render_widget(gauge, area);
}

fn stage_progress_weight(stage: Option<&str>) -> f64 {
	match stage {
		Some("loading") => 0.05,
		Some("loading model") => 0.10,
		Some("estimating depth") => 0.50,
		Some("saving depth") => 0.65,
		Some("generating stereo") => 0.80,
		Some("saving") => 0.90,
		Some("scanning") => 0.10,
		Some("extracting") => 0.15,
		Some("processing") => 0.70,
		Some("encoding") => 0.85,
		Some("packaging") => 0.95,
		_ => 0.0,
	}
}

fn draw_files(frame: &mut Frame, area: Rect, state: &AppState) {
	if area.height == 0 {
		return;
	}

	let active_files: Vec<(usize, &FileState)> = state
		.files
		.iter()
		.enumerate()
		.filter(|(_, f)| matches!(f.status, FileStatus::Processing | FileStatus::Pending))
		.collect();

	let mut y = area.y;
	
	for (i, f) in active_files.iter() {
		if y >= area.y + area.height {
			break;
		}

		let line_area = Rect::new(area.x, y, area.width, 1);
		let line = file_line(f, *i, state);
		frame.render_widget(Paragraph::new(line), line_area);
		y += 1;

		if f.status == FileStatus::Processing {
			if let Some(stage_line_content) = stage_line(f, state.avg_depth_time) {
				if y < area.y + area.height {
					let stage_area = Rect::new(area.x, y, area.width, 1);
					frame.render_widget(Paragraph::new(stage_line_content), stage_area);
					y += 1;
				}
			}
		}
	}
}

fn file_line<'a>(f: &FileState, _index: usize, _state: &AppState) -> Line<'a> {
	match &f.status {
		FileStatus::Pending => Line::from(vec![
			Span::styled("  ", Style::default().fg(Color::DarkGray)),
			Span::styled(f.filename.clone(), Style::default().fg(Color::DarkGray)),
		]),
		FileStatus::Processing => {
			let icon = "⠒ ";
			Line::from(vec![
				Span::styled(icon, Style::default().fg(Color::Cyan)),
				Span::styled(f.filename.clone(), Style::default().bold()),
			])
		}
		FileStatus::Done { outputs, duration } => {
			let secs = duration.as_secs_f64();
			let time_str = if secs >= 60.0 {
				format!("{}m{:02.0}s", secs as u64 / 60, secs % 60.0)
			} else {
				format!("{:.1}s", secs)
			};
			let mut spans = vec![
				Span::styled("✔ ", Style::default().fg(Color::Green).bold()),
				Span::raw(f.filename.clone()),
			];
			if !outputs.is_empty() {
				spans.push(Span::styled(
					format!("  → {}", outputs.join(", ")),
					Style::default().fg(Color::DarkGray),
				));
			}
			spans.push(Span::styled(
				format!("  ({})", time_str),
				Style::default().fg(Color::DarkGray),
			));
			Line::from(spans)
		}
		FileStatus::Error(msg) => Line::from(vec![
			Span::styled("✗ ", Style::default().fg(Color::Red).bold()),
			Span::raw(f.filename.clone()),
			Span::styled(format!("  {}", msg), Style::default().fg(Color::Red)),
		]),
	}
}

fn stage_line<'a>(f: &FileState, _avg_depth_time: Option<f64>) -> Option<Line<'a>> {
	if f.status != FileStatus::Processing {
		return None;
	}

	let stage = f.stage.as_ref()?;
	let elapsed = f.stage_started_at.as_ref()?.elapsed().as_secs_f64();

	let mut spans = vec![
		Span::raw("  "),
		Span::styled(stage.clone(), Style::default().fg(Color::Yellow)),
		Span::raw("  "),
	];

	let bar_width: usize = 15;
	let bar = match stage.as_str() {
		"estimating depth" => {
			if let Some(avg) = _avg_depth_time {
				let ratio = (elapsed / avg).min(1.0);
				let filled = (ratio * bar_width as f64) as usize;
				let bar_chars: String = "━".repeat(filled) + "╸" + &"─".repeat(bar_width.saturating_sub(filled + 1));
				format!("{} {:.1}s / ~{:.1}s", bar_chars, elapsed, avg)
			} else {
				format!("{:.1}s", elapsed)
			}
		}
		"generating stereo" => {
			let ratio = f.stage_progress / 100.0;
			let filled = (ratio * bar_width as f64) as usize;
			let bar_chars: String = "━".repeat(filled) + "╸" + &"─".repeat(bar_width.saturating_sub(filled + 1));
			format!("{} {:.0}%", bar_chars, f.stage_progress)
		}
		"processing" if f.video_total > 0 => {
			let ratio = f.stage_progress / 100.0;
			let filled = (ratio * bar_width as f64) as usize;
			let bar_chars: String = "━".repeat(filled) + "╸" + &"─".repeat(bar_width.saturating_sub(filled + 1));
			format!(
				"{} {:>5.1} fps {}/{} eta {}",
				bar_chars, f.video_fps, f.video_frame, f.video_total, f.video_eta
			)
		}
		_ => format!("{:.1}s", elapsed),
	};

	spans.push(Span::styled(bar, Style::default().fg(Color::Cyan)));
	Some(Line::from(spans))
}

pub fn init_terminal() -> io::Result<Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>> {
	enable_raw_mode()?;
	let backend = ratatui::backend::CrosstermBackend::new(stdout());
	let terminal = Terminal::with_options(
		backend,
		TerminalOptions {
			viewport: Viewport::Inline(8),
		},
	)?;
	Ok(terminal)
}

pub fn restore_terminal() {
	let _ = disable_raw_mode();
	let _ = execute!(stdout(), crossterm::cursor::Show);
}

pub fn render_frame(
	terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
	state: &AppState,
) -> io::Result<()> {
	terminal.draw(|frame| draw(frame, state))?;
	Ok(())
}

pub fn insert_completed_line(
	terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
	f: &FileState,
	index: usize,
	state: &AppState,
) -> io::Result<()> {
	let line = file_line(f, index, state);
	terminal.insert_before(1, |buf| {
		Paragraph::new(line).render(buf.area, buf);
	})?;
	Ok(())
}
