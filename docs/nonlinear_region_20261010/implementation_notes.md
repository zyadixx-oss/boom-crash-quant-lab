Before freezing or fitting any historical model, the existing-cache precheck
found that pandas parsed saved planned_end strings into datetime64[us, UTC],
while the exact reconstructed horizon used datetime64[ns, UTC]. The numerical
timestamps were unchanged. The new reader now explicitly normalizes this one
column to nanoseconds before the strict equality check; a regression fixture
covers that round trip. No saved labels, events, prices or old source files
were edited, and no fit/output was produced by the failed precheck.
