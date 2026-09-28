-- Render the deck to PDF through PowerPoint itself, which is the renderer the
-- deck will actually be read in -- so font metrics and text fit in QA are the
-- real ones rather than a substitute's.
--
-- Only the presentation this script opens is touched: it is identified by name
-- rather than by "active presentation", so another deck the user has open is
-- never closed, and nothing is ever saved over.

on run argv
	set srcPath to item 1 of argv
	set outPath to item 2 of argv

	tell application "Microsoft PowerPoint"
		open (POSIX file srcPath)
		delay 1
		set theDoc to presentation 1
		set docName to name of theDoc
		save theDoc in (POSIX file outPath) as save as PDF
		delay 1
		close presentation docName saving no
	end tell
	return "ok"
end run
