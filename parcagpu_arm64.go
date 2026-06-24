package parcagpu

import "embed"

//go:embed archive/arm64
var archive embed.FS

const archivePrefix = `archive/arm64`
