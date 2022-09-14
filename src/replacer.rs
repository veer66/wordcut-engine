use regex::Regex;
use thiserror::Error;

#[allow(dead_code)]
#[derive(Error, Debug)]
pub enum ReplacerError {
    #[error("Cannot create immediate rule `{0}` because `{1}`")]
    CannotCreateImmRule(String, String),
}

#[derive(Deserialize, Debug, Clone)]
pub struct Rule {
    pub pattern: String,
    pub replacement: String,
}

#[derive(Debug, Clone)]
pub struct ImmRule {
    pub pattern: Regex,
    pub replacement: String,
}

impl ImmRule {
    #[allow(dead_code)]
    pub fn from_rule(rule: &Rule) -> Result<ImmRule, ReplacerError> {
        let pattern = Regex::new(&rule.pattern).map_err(|e| {
            ReplacerError::CannotCreateImmRule(rule.pattern.clone(), format!("{}", e))
        })?;
        Ok(ImmRule {
            pattern,
            replacement: rule.replacement.clone(),
        })
    }

    #[allow(dead_code)]
    pub fn from_rules(rules: &[Rule]) -> Result<Vec<ImmRule>, ReplacerError> {
        let mut imm_rules = Vec::new();
        for rule in rules {
            let imm_rule = Self::from_rule(rule)?;
            imm_rules.push(imm_rule)
        }
        Ok(imm_rules)
    }
}

#[allow(dead_code)]
pub fn replace(rules: &[ImmRule], text: &str) -> String {
    if rules.len() == 0 {
	return text.to_string()
    }
    let mut mod_text = text.to_string();
    for rule in rules {
	mod_text = rule.pattern.replace(&text, &rule.replacement).to_string();
    }
    return mod_text
}

#[cfg(test)]
mod tests {
    extern crate serde_json;
    use super::*;

    #[test]
    fn sara_am() {
	let rule = r###"{"pattern": "ํา", "replacement": "ำ"}"###;
	let rule: Rule = serde_json::from_str(rule).unwrap();
	let imm_rules = ImmRule::from_rules(&vec![rule]).unwrap();
	let mod_text = replace(&imm_rules, "สําหรับข้อเสนอ");
	assert_eq!(mod_text, "สำหรับข้อเสนอ");
    }
}
