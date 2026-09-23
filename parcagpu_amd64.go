package parcagpu

import "embed"

//go:embed archive/amd64
var archive embed.FS

const archivePrefix = `archive/amd64`
