# PMO program fixture

`pmo_armc_panel_v1.json` contains 80 saved executable programs and source
states used to test created-atom rebinding. It carries no oracle scores. The
four delete-then-insert witnesses are first so a bounded test exercises that
failure mode.

The fixture was copied byte-for-byte from commit
`a738a0eb5b0cbe4f7b303d5918baffb5021a2802` on the PMO experiment branch.
Its SHA-256 is
`f4fe5ffa57a6bc94022a0d104c39b4216b54610d2c32698a14dc6ccfc4b2fc27`.
Tests fail if these bytes are missing or changed.
